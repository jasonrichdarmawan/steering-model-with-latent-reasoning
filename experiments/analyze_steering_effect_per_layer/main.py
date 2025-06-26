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
  import sys
  from importlib import reload

  print("Reloading modules to ensure the latest code is used.")
  
  reload(sys.modules.get('asepl_utils.parse_args', sys))
  reload(sys.modules.get('asepl_utils.set_save_grad_hooks', sys))
  reload(sys.modules.get('asepl_utils', sys))

  reload(sys.modules.get('utils.use_deterministic_algorithms', sys))
  reload(sys.modules.get('utils.load_model_and_tokenizer', sys))
  reload(sys.modules.get('utils.load_json_dataset', sys))
  reload(sys.modules.get('utils.load_hidden_states_cache', sys))
  reload(sys.modules.get('utils.compute_candidate_directions', sys))
  reload(sys.modules.get('utils.prepare_queries', sys))
  reload(sys.modules.get('utils.get_n_layers', sys))
  reload(sys.modules.get('utils', sys))

from asepl_utils import parse_args
from asepl_utils import set_save_grad_hooks

from utils import use_deterministic_algorithms
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import load_hidden_states_cache
from utils import compute_candidate_directions
from utils import prepare_queries
from utils import get_n_layers
from utils import ProjectionHookConfig
from utils import set_activations_hooks

import random
from tqdm import tqdm
import torch
from jaxtyping import Float
from torch.nn.functional import log_softmax, kl_div
from torch import Tensor
import gc

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/home/npu-tao/jason"
  sys.argv = [
    'main.py',

    '--models_path', f'{root_path}/transformers',
    '--model_name', 'huginn-0125',

    '--hidden_states_cache_file_path', f'{root_path}/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt',
    '--data_path', f'{root_path}/datasets/lirefs',
    '--data_name', 'mmlu-pro-3000samples.json',
    '--data_sample_size', '24',
    '--data_batch_size', '1',

    '--huginn_num_steps', '32',

    '--with_post_hook',
    '--projection_scale', '0.1',

    '--output_file_path', f'{root_path}/experiments/analyze_steering_effect_per_layer/huginn-0125.json',
  ]

args = parse_args()
print("Parsed arguments:")
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Using deterministic algorithms for reproducibility.")
use_deterministic_algorithms()

# %%

print("Loading model and tokenizer.")
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

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
  device_map=device_map,
)

print(f"Model device map:")
print(model.hf_device_map)

# %%

file_path = os.path.join(
  args['data_path'],
  args['data_name'],
)

print(f"Loading dataset from file: {file_path}")
data = load_json_dataset(
  file_path=file_path,
)

print("Determining reasoning and memorizing indices based on memory_reason_score.")
reasoning_indices = [
  index for index, sample in enumerate(data)
  if sample["memory_reason_score"] > 0.5
]

memorizing_indices = [
  index for index, sample in enumerate(data)
  if sample["memory_reason_score"] <= 0.5
]

# %%

print("Loading hidden states cache from file.")
hidden_states_cache = load_hidden_states_cache(
  file_path=args['hidden_states_cache_file_path'],
)

# %%

print("Computing candidate directions based on hidden states cache.")
candidate_directions = compute_candidate_directions(
  model=model,
  hidden_states_cache=hidden_states_cache,
  reasoning_indices=reasoning_indices,
  memorizing_indices=memorizing_indices,
)

# %%

if args['data_sample_size']:
  print(f"Sampling {args['data_sample_size']} samples from the dataset.")
  sampled_data = random.sample(
    data, args['data_sample_size']
  )
else:
  print("No data sample size specified, using the entire dataset.")
  sampled_data = data

# %%

print("Preparing queries for the model.")
queries = prepare_queries(
  model_name=model.config.model_type,
  data=sampled_data,
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

n_layers = get_n_layers(model)

print("Setting up activations hooks for the model.")
projection_hook_config = ProjectionHookConfig(
  layer_indices=range(n_layers),
  candidate_directions=candidate_directions,
  pre_hook=args["with_pre_hook"],
  post_hook=args["with_post_hook"],
  scale=args["projection_scale"],
)
activations_hooks = set_activations_hooks(
  model=model,
  candidate_directions=candidate_directions,
  config=projection_hook_config,
)

# %%

print("Setting up save_grad hooks for the model.")
save_grad_hooks, gradients = set_save_grad_hooks(
  model=model,
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
  case name if name.startswith("huginn_"):
    effects = [0 for _ in range(n_layers)]

    for queries_batch in tqdm(queries_batched):
      input_ids = tokenizer(
        queries_batch,
        return_tensors='pt',
        add_special_tokens=False,
        padding='longest',
        return_token_type_ids=False,
      ).input_ids.to(device=model.device)

      outputs = model(
        input_ids=input_ids,
        num_steps=(0, args["huginn_num_steps"]),
        output_details={
          "return_logits": True,
          "return_latents": False,
          "return_attention": False,
          "return_head": False,
          "return_stats": False,
        }
      )

      loss = compute_kl_divergence(outputs["logits"])
      loss.backward()

      for layer_index in range(n_layers):
        effect = (
          gradients[layer_index]
          @ candidate_directions[layer_index]
        ).mean().abs()
        effects[layer_index] += effect.item()

      del input_ids
      del outputs
      del loss
      model.zero_grad()
      gradients.clear()
      gc.collect()
      torch.cuda.empty_cache()
  case _:
    raise ValueError(f"Model type {model.config.model_type} is not supported.")

print("Effects per layer:")
print(effects)

# %%

print("Removing hooks from the model.")
for activations_hook in activations_hooks:
  activations_hook.remove()
for save_grad_hook in save_grad_hooks:
  save_grad_hook.remove()

# %%

if args['output_file_path'] is None:
  print("No output path specified. Results will not be saved.")
  sys.exit(0)

try:
  print(f"Loading existing results from: {args['output_file_path']}")
  output = torch.load(
    args["output_file_path"],
    map_location='cpu',
    weights_only=False,
  )
except FileNotFoundError:
  print(f"File not found: {args['output_file_path']}. Creating a new results dictionary.")
  output = {}

# %%

output_key = ' '.join(
  [
    f"{key}={value}"
    for key, value in args.items()
    if key in [
      'model_name',

      'data_name',
      'data_sample_size',

      'huginn_num_steps',

      'with_pre_hook',
      'with_post_hook',
      'projection_scale',
    ]
  ]
)
print(f"Using output key: {output_key}")

# %%

output[output_key] = effects

# %%

output_path = os.path.dirname(args['output_file_path'])
os.makedirs(output_path, exist_ok=True)

print(f"Saving evaluation results to: {args['output_file_path']}")
torch.save(output, args['output_file_path'])

# %%