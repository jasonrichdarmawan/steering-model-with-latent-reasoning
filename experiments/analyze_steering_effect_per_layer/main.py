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

from utils import DirectionNormalizationMode
from utils import enable_reproducibility
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import get_device_map
from utils import prepare_queries
from utils import tokenize_text

from tqdm import tqdm
import torch as t
from torch import Tensor
from torch.nn.functional import log_softmax, kl_div
from jaxtyping import Float
import numpy as np
import math
import gc
import matplotlib.pyplot as plt

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  MODEL_NAME = "huginn-0125"
  DIRECTION_NORMALIZATION_MODE = str(DirectionNormalizationMode.UNIT_VECTOR)
  sys.argv = [
    'main.py',

    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    # '--device', 'cuda:0',

    '--huginn_num_steps', '32',

    '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/candidate_directions_FIRST_ANSWER_TOKEN.pt',
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
    device_map = args['device'] or {
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
    device_map = args['device'] or "cuda:0"
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
candidate_directions = t.load(
  args['candidate_directions_file_path'],
  map_location='cpu',
  weights_only=False,
)[args['model_name']]

directions: dict[int, Float[Tensor, "n_embd"]] = {}
for layer_index in range(len(candidate_directions['reasoning'])):
  directions[layer_index] = candidate_directions['reasoning'][layer_index] - candidate_directions['memorizing'][layer_index]

directions_normalized = {
  layer_index: direction / direction.norm(dim=-1, keepdim=True)
  for layer_index, direction in directions.items()
}

device_map = get_device_map(model=model)
for layer_index, direction in directions_normalized.items():
  directions_normalized[layer_index] = direction.to(
    dtype=model.dtype,
    device=device_map[layer_index]
  )

del candidate_directions
del directions

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

print(f"Splitting the data indices into reasoning and memorizing indices")
reasoning_indices = [
  index for index, sample in enumerate(data)
  if sample['memory_reason_score'] > 0.5
]
memorizing_indices = [
  index for index, sample in enumerate(data)
  if sample['memory_reason_score'] <= 0.5
]

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

del data

# %%

print("Sorting queries by query length")
queries_with_idx = list(enumerate(queries))
queries_with_label = []
for index, query in queries_with_idx:
  if index in reasoning_indices:
    label = 'reasoning'
  elif index in memorizing_indices:
    label = 'memorizing'
  else:
    raise ValueError(f"Index {index} not found in either reasoning or memorizing indices.")
  queries_with_label.append({
      'label': label,
      'query': query,
    })
queries_sorted = sorted(
  queries_with_label,
  key=lambda item: len(item['query']),
)

del queries_with_idx
del queries_with_label

# %%

print("Preparing entries for batching")
queries_batched = [
  queries_sorted[i:i + args["data_batch_size"]]
  for i in range(
    0, len(queries_sorted), args["data_batch_size"]
  )
  if len(queries_sorted[i:i + args["data_batch_size"]]) == args["data_batch_size"]
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

effects: dict[
  str, 
  dict[int, list[float]]
] = {
  "reasoning set": {},
  "memorizing set": {},
  "overall set": {}
}

match model.config.model_type:
  case "huginn_raven":
    for queries_batch in tqdm(queries_batched):
      inputs = tokenize_text(
        model=model,
        tokenizer=tokenizer,
        text=[item['query'] for item in queries_batch],
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

      labels_batch = np.array(
        [item['label'] for item in queries_batch]
      )

      reasoning_indices_batch = np.where(
        labels_batch == 'reasoning'
      )[0]
      memorizing_indies_batch = np.where(
        labels_batch == 'memorizing'
      )[0]

      for layer_index in list(
        gradients.keys() & directions_normalized.keys()
      ):
        effects_tensor = (
          gradients[layer_index]
          @ directions_normalized[layer_index]
        ) # shape (batch, seq_len)

        reasoning_effects_batch = effects_tensor[reasoning_indices_batch].mean().abs().item()
        memorizing_effects_batch = effects_tensor[memorizing_indies_batch].mean().abs().item()
        overall_effects_batch = effects_tensor.mean().abs().item()

        effects_sets = {
          "reasoning set": reasoning_effects_batch,
          "memorizing set": memorizing_effects_batch,
          "overall set": overall_effects_batch,
        }
        for set_name, value in effects_sets.items():
          if effects[set_name].get(layer_index) is None:
            effects[set_name][layer_index] = []
          if not math.isnan(value):
            effects[set_name][layer_index].append(value)

      del inputs
      del outputs
      del loss
      model.zero_grad()
      gradients.clear()
      gc.collect()
      t.cuda.empty_cache()
  case _:
    raise ValueError(f"Model type {model.config.model_type} is not supported.")

print("Removing hooks from the model.")
for save_grad_hook in save_grad_hooks:
  save_grad_hook.remove()

# %%

print("Sort layer index by the mean effect value across all batches.")
top_effects = {
  "reasoning set": [],
  "memorizing set": [],
  "overall set": [],
}

for set_name, set_effects in effects.items():
  top_effects[set_name] = sorted(
    set_effects.items(),
    key=lambda item: np.mean(item[1]),
    reverse=True,
  )

for set_name, set_effects in top_effects.items():
  print(f"Top effects for {set_name}:")
  for layer_index, effect_values in set_effects:
    mean_effect = np.mean(effect_values)
    std_effect = np.std(effect_values)
    print(
      f"Layer {layer_index}: "
      f"effect={mean_effect:.4f} ± {std_effect:.4f}, "
    )

# %%

"""
High cosine similarity between directions 
and model embedding can indicate 
that the model retains 
token representation information
in early layers rather than behavioral patterns. 

Reference: https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/train-steering-vectors/cosine_sim.py
"""

print("Compute cosine similarities between directions and model embedding layers.")
cosine_similarities = {}
module = model.transformer.wte
cloned_directions = {
  layer_index: direction.detach().clone().to(device=module.weight.device)
  for layer_index, direction in directions_normalized.items()
}
for layer_index, direction in cloned_directions.items():
  cosine_similarities[layer_index] = (
    direction @ model.transformer.wte.weight.data.T
  ).max().item()

print("Sort cosine similarities by value.")
top_cosine_similarities = sorted(
  cosine_similarities.items(),
  key=lambda item: item[1],
  reverse=True,
)
for layer_index, cosine_similarity in top_cosine_similarities:
  print(f"Layer {layer_index}: {cosine_similarity:.4f}")

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
  output = t.load(
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

      'direction_normalization_mode',
    ]
  ]
)
print(f"Using output key: {output_key}")
output.setdefault(output_key, {})
output[output_key].update({
  "top_effects": top_effects,
  "top_cosine_similarities": top_cosine_similarities,
})

# %%

os.makedirs(args['output_path'], exist_ok=True)

print(f"Saving evaluation results to: {output_file_path}")
t.save(output, output_file_path)

# %%

print("Creating effect and cosine similarity per layer plot")
nrows = 1
ncols = 2
fig, axes = plt.subplots(
  nrows=nrows, 
  ncols=ncols, 
  figsize=(6.4 * ncols, 4.8 * nrows)
)
axes = axes.flatten()

axes[0].plot(
  range(len(effects['reasoning set'])),
  [
    np.mean(effect_values) 
    for _, effect_values 
    in effects['reasoning set'].items()
  ],
  label='Reasoning Set',
)

axes[0].plot(
  range(len(effects['memorizing set'])),
  [
    np.mean(effect_values) 
    for _, effect_values 
    in effects['memorizing set'].items()
  ],
  label='Memorizing Set',
)

axes[0].plot(
  range(len(effects['overall set'])),
  [
    np.mean(effect_values) 
    for _, effect_values 
    in effects['overall set'].items()
  ],
  label='Overall Set',
)

axes[0].set_title('Mean Gradients @ Direction per Layer')
axes[0].set_xlabel('Layer Index')
axes[0].set_ylabel('Mean Gradients @ Direction Value')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
  range(len(cosine_similarities)),
  [
    cosine_similarity 
    for _, cosine_similarity 
    in cosine_similarities.items()
  ],
)
axes[1].set_title('Embedding Weights @ Direction per Layer Cosine Similarity')
axes[1].set_xlabel('Layer Index')
axes[1].set_ylabel('Cosine Similarity Value')
axes[1].grid(True)

fig.suptitle(args['model_name'])

# %%

output_plot_path = os.path.join(
  args['output_path'],
  f"{args['model_name']}_effects_and_cosine_similarities.pdf"
)
print(f"Saving the effect per layer plot to: {output_plot_path}")
fig.savefig(fname=output_plot_path)

# %%
