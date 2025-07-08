# %%

import os
import sys

# To be able to import modules from the utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")
  sys.path.insert(0, project_root)

# %%

if False:
  from utils import reload_modules
  
  reload_modules(
    project_root=project_root,
  )

from asepl_utils import parse_args
from asepl_utils import GradientMode
from asepl_utils import set_save_grad_hooks
from asepl_utils import create_effect_per_layer_plot
from asepl_utils import compute_cosine_similarities
from asepl_utils import create_cosine_similarity_plot

from utils import ProcessHiddenStatesMode
from utils import DirectionNormalizationMode
from utils import enable_reproducibility
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import compute_directions
from utils import prepare_queries
from utils import tokenize_text

from tqdm import tqdm
import torch
from torch import Tensor
from torch.nn.functional import log_softmax, kl_div
from jaxtyping import Float
import gc

import numpy as np

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  MODEL_NAME = "huginn-0125"
  PROCESS_HIDDEN_STATES_MODE = str(ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN)
  DIRECTION_NORMALIZATION_MODE = str(DirectionNormalizationMode.UNIT_VECTOR)
  sys.argv = [
    'main.py',

    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    '--huginn_num_steps', '32',

    '--process_hidden_states_mode', PROCESS_HIDDEN_STATES_MODE,
    '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/{MODEL_NAME}_mmlu-pro-3000samples.json_{PROCESS_HIDDEN_STATES_MODE}_candidate_directions.pt',
    '--direction_normalization_mode', DIRECTION_NORMALIZATION_MODE,

    '--data_path', f'{WORKSPACE_PATH}/datasets/lirefs',
    '--data_name', 'mmlu-pro-3000samples.json',
    '--data_sample_size', '24',
    '--data_batch_size', '1',

    '--output_path', f'{WORKSPACE_PATH}/experiments/analyze_steering_effect_per_layer/{MODEL_NAME}',
  ]

args = parse_args()
print("Parsed arguments:")
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Using deterministic algorithms for reproducibility.")
enable_reproducibility()

# %%

print("Loading model and tokenizer.")
match args["model_name"]:
  case "huginn-0125":
    device_map = {
      "transformer.wte": 0,
      "freqs_cis": 0,
      "transformer.prelude": 0,
      "transformer.adapter": 0,
      "transformer.core_block.0": 0,
      "transformer.core_block.1": 0,
      "transformer.core_block.2": 1,
      "transformer.core_block.3": 1,
      "transformer.coda": 1,
      "transformer.ln_f": 1,
      "lm_head": 0,
    }
  case "Meta-Llama-3-8B":
    device_map = "cuda:0"
  case _:
    raise ValueError(f"Model type {args['model_name']} is not supported for loading.")

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
  device_map=device_map,
)

print(f"Model device map:")
print(model.hf_device_map)

# %%

print(f"Compute directions from file: {args['candidate_directions_file_path']}")
candidate_directions = torch.load(
  args['candidate_directions_file_path'],
  map_location='cpu',
  weights_only=False,
)
directions = compute_directions(
  model=model,
  candidate_directions=candidate_directions,
  positive_label="reasoning",
  negative_label="memorizing",
  overall_label="overall",
  normalization_mode=args['direction_normalization_mode'],
)

del candidate_directions

# %%

file_path = os.path.join(
  args['data_path'],
  args['data_name'],
)

print(f"Loading dataset from JSON file with sample size {args['data_sample_size']}: {file_path}")
data = load_json_dataset(
  file_path=file_path,
  sample_size=args['data_sample_size'],
)

# %%

print("Preparing queries for the model.")
queries = prepare_queries(
  model_name=model.config.model_type,
  data=data,
  data_name=args['data_name'],
  tokenizer=tokenizer,
  apply_chat_template=False,
  system_prompt=None,
  fewshot_prompts=None,
  with_cot=False,
)

queries_batched = [
  queries[i:i + args["data_batch_size"]]
  for i in range(
    0, len(queries), args["data_batch_size"]
  )
]

# %%

print("Setting up save_grad hooks for the model with gradient_mode last_token.")
save_grad_hooks, gradients = set_save_grad_hooks(
  model=model,
  gradient_mode=GradientMode.LAST_TOKEN,
)

# %%

print("Computing the effect of each candidate direction per layer.")
effect_per_layer: list[float] = []

def compute_kl_divergence(logits: Float[Tensor, "batch seq_len n_embd"]):
  # Reference: https://github.com/jasonrichdarmawan/steering-thinking-models/blob/ec310e0fe1b132093f7c44a8d2e39f173d2f75ae/vector-layer-attribution/analyze_layer_effects.py
  probs = log_softmax(logits, dim=-1)
  detached_probs = log_softmax(logits.detach(), dim=-1)
  return kl_div(probs, detached_probs, reduction="batchmean")

match model.config.model_type:
  case "huginn_raven":
    effects: dict[int, list[float]] = {}

    for queries_batch in tqdm(queries_batched):
      inputs = tokenize_text(
        model=model,
        tokenizer=tokenizer,
        text=queries_batch,
      )

      outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_steps=(0, args["huginn_num_steps"]),
        output_details={
          "return_logits": True,
          "return_latents": False,
          "return_attention": False,
          "return_head": False,
          "return_stats": False,
        }
      )

      loss = compute_kl_divergence(outputs["logits"][:, -1:])
      loss.backward()

      for layer_index in list(
        gradients.keys() & directions.keys()
      ):
        effect = (
          gradients[layer_index]
          @ directions[layer_index]
        ).mean().abs()
        if effects.get(layer_index) is None:
          effects[layer_index] = []
        effects[layer_index].append(effect.item())

      del inputs
      del outputs
      del loss
      model.zero_grad()
      gradients.clear()
      gc.collect()
      torch.cuda.empty_cache()
  case _:
    raise ValueError(f"Model type {model.config.model_type} is not supported.")

# %%

print("Effects per layer:")
print(effects)

# %%

print("Layers by the highest effects:")
top_effects = sorted(
  effects.items(),
  key=lambda item: np.mean(item[1]),
  reverse=True,
)
for layer_index, effect_values in top_effects:
  print(f"Layer {layer_index}: {np.mean(effect_values):.4f} ± {np.std(effect_values):.4f} (n={len(effect_values)})")

# %%

print("Removing hooks from the model.")
for save_grad_hook in save_grad_hooks:
  save_grad_hook.remove()

# %%

if args['output_path'] is None:
  print("No output path specified. Results will not be saved.")
  sys.exit(0)

try:
  output_file_path = os.path.join(
    args['output_path'],
    f"{args['model_name']}.json"
  )
  print(f"Loading existing results from: {output_file_path}")
  output = torch.load(
    output_file_path,
    map_location='cpu',
    weights_only=False,
  )
except FileNotFoundError:
  print(f"File not found: {output_file_path}. Creating a new results dictionary.")
  output = {}

# %%

output_key = ' '.join(
  [
    f"{key}={value}"
    for key, value in args.items()
    if key in [
      'model_name',

      'huginn_num_steps',

      'data_name',
      'data_sample_size',

      'process_hidden_states_mode',
      'direction_normalization_mode',
    ]
  ]
)
print(f"Using output key: {output_key}")

# %%

output[output_key] = effects

# %%

os.makedirs(args['output_path'], exist_ok=True)

print(f"Saving evaluation results to: {output_file_path}")
torch.save(output, output_file_path)

# %%

fig = create_effect_per_layer_plot(
  effects=effects,
  model_name=args['model_name'],
)

# %%

output_plot_path = os.path.join(
  args['output_path'],
  f"{args['model_name']}_{args['process_hidden_states_mode']}_{args['direction_normalization_mode']}_effect_per_layer.pdf"
)
print(f"Saving the effect per layer plot to: {output_plot_path}")
fig.savefig(
  fname=output_plot_path,
  dpi=300,
  bbox_inches='tight',
  facecolor='white',
  edgecolor='none',
)

# %%

def process(
  x1: dict[int, Float[Tensor, "n_embd"]],
  x2: Float[Tensor, "block_size n_embd"],
  x2_name: str,
  fig_file_path: str | None = None,
):
  """
  High cosine similarity between candidate directions 
  and model embedding or unembedding layers can indicate 
  that the model retains token representation information
  in early layers rather than behavioral patterns. 
  In other words, the effect value computed in these layers
  can be misleading.

  Reference: https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/train-steering-vectors/cosine_sim.py
  """
  cosine_similarities = compute_cosine_similarities(
    x1=x1,
    x2=x2,
  )

  print(f"Layers by the highest cosine similarity between candidate directions and {x2_name}:")
  top_cosine_similarities = sorted(
    cosine_similarities.items(),
    key=lambda item: item[1],
    reverse=True,
  )
  for layer_index, cosine_similarity in top_cosine_similarities:
    print(f"Layer {layer_index}: {cosine_similarity:.4f}")

  cosine_similarities_fig = create_cosine_similarity_plot(
    cosine_similarities=cosine_similarities,
    x2_name=x2_name,
  )

  if fig_file_path is None:
    print("No figure file path specified. Plot will not be saved.")
    return
  
  print(f"Saving the cosine similarity between candidate directions and {x2_name} plot to: {output_file_path}")
  cosine_similarities_fig.savefig(
    fname=fig_file_path,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
  )

  return cosine_similarities

match model.config.model_type:
  case "huginn_raven":
    cosine_similarities = process(
      x1=directions,
      x2=model.transformer.wte.weight.data,
      x2_name="embedding layer weight",
      fig_file_path=os.path.join(
        args['output_path'],
        f"{args['model_name']}_{args['process_hidden_states_mode']}_{args['direction_normalization_mode']}_embed_cosine_similarities.pdf"
      ),
    )
  case _:
    raise ValueError(f"Model type {model.config.model_type} is not supported for cosine similarity analysis.")

# %%

for layer_index, effect_values in top_effects:
  mean_effect = np.mean(effect_values)
  std_effect = np.std(effect_values)
  similarity = cosine_similarities[layer_index]
  print(
    f"Layer {layer_index}: "
    f"effect={mean_effect:.4f} ± {std_effect:.4f}, "
    f"cosine similarity={similarity:.4f}"
  )

# %%
