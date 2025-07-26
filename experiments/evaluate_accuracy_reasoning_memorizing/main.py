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

from earm_utils import parse_args

from utils import enable_reproducibility

from utils import load_model_and_tokenizer

from utils import load_json_dataset

from utils import prepare_fewshot_prompts
from utils import prepare_queries

from utils import get_device_map
from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode
from utils import ProjectionHookConfig
from utils import ProjectionHookConfigLiReFs
from utils import set_activations_hooks

from utils import tokenize_text
from utils import generate
from utils import set_model_predict_correctness

import torch as t
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  
  MODEL_NAME = "Meta-Llama-3-8B"
  
  DIRECTION_NORMALIZATION_MODE = DirectionNormalizationMode.UNIT_VECTOR
  PROJECTION_HOOK_MODE = ProjectionHookMode.FEATURE_ADDITION
  MODIFICATION_MODE = TokenModificationMode.ALL_TOKENS
  
  sys.argv = [
    'main.py',
    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    '--device', 'cuda:0',

    # '--huginn_num_steps', '129',

    '--test_data_path', f'{WORKSPACE_PATH}/datasets/lirefs',
    '--test_data_name', 'mmlu-pro-3000samples.json',
    # '--test_data_sample_size', '24',
    '--with_fewshot_prompts',
    # '--with_cot',
    '--batch_size', '1',

    '--with_intervention',
    
    '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/candidate_directions_FIRST_ANSWER_TOKEN.pt',
    
    '--layer_indices', '8',
    '--direction_normalization_mode', str(DIRECTION_NORMALIZATION_MODE),
    '--projection_hook_mode', str(PROJECTION_HOOK_MODE),
    '--modification_mode', str(MODIFICATION_MODE),
    '--with_hidden_states_pre_hook',
    # '--with_hidden_states_post_hook',
    '--scale', '0.1',

    '--output_file_path', f'{WORKSPACE_PATH}/experiments/reasoning_memorizing_accuracy/Meta-Llama-3-8B.json'
  ]

args = parse_args()
print(f"Parsed arguments:")
print('#' * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Setting deterministic algorithms for reproducibility.")
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
    device_map = args['device'] or {
      "model.embed_tokens": 0,
      "model.layers.0": 0,
      "model.layers.1": 0,
      "model.layers.2": 0,
      "model.layers.3": 0,
      "model.layers.4": 0,
      "model.layers.5": 0,
      "model.layers.6": 0,
      "model.layers.7": 0,
      "model.layers.8": 0,
      "model.layers.9": 0,
      "model.layers.10": 0,
      "model.layers.11": 0,
      "model.layers.12": 0,
      "model.layers.13": 1,
      "model.layers.14": 1,
      "model.layers.15": 1,
      "model.layers.16": 1,
      "model.layers.17": 1,
      "model.layers.18": 1,
      "model.layers.19": 1,
      "model.layers.20": 1,
      "model.layers.21": 1,
      "model.layers.22": 1,
      "model.layers.23": 1,
      "model.layers.24": 1,
      "model.layers.25": 1,
      "model.layers.26": 1,
      "model.layers.27": 1,
      "model.layers.28": 1,
      "model.layers.29": 1,
      "model.layers.30": 1,
      "model.layers.31": 1,
      "model.norm": 1,
      "model.rotary_emb": 0,
      "lm_head": 0,
    }
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

from transformers.models.llama.modeling_llama import LlamaForCausalLM

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
  device_map=device_map,
)

# %%

match args['test_data_name']:
  case "mmlu-pro-3000samples.json":
    file_path = os.path.join(
      args['test_data_path'],
      "mmlu-pro-3000samples.json"
    )
    test_dataset = load_json_dataset(
      file_path=file_path,
      sample_size=args['test_data_sample_size'],
    )
    print(f"Loaded test dataset from {file_path} with {len(test_dataset)} samples.")
  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%

if args['with_fewshot_prompts']:
  fewshot_prompts = prepare_fewshot_prompts(
    data_path=args['test_data_path'],
    data_name=args['test_data_name'],
    with_cot=args['with_cot'],
  )
  print(f"Prepared few-shot prompts for {args['test_data_name']} with {len(fewshot_prompts)} categories.")
else:
  print("No few-shot prompts will be used for the test dataset.")
  fewshot_prompts = None

# %%

print("Preparing queries for the model.")
queries = prepare_queries(
  model_name=model.config.model_type,
  data=test_dataset,
  data_name=args['test_data_name'],
  tokenizer=tokenizer,
  apply_chat_template=False,
  fewshot_prompts=fewshot_prompts,
  with_cot=args['with_cot'],
)

# %%

print(f"Splitting the sampled data indices into reasoning and memorizing indices")
reasoning_indices = [
  index for index, sample in enumerate(test_dataset) 
  if sample['memory_reason_score'] > 0.5
]
memorizing_indices = [
  index for index, sample in enumerate(test_dataset) 
  if sample['memory_reason_score'] <= 0.5
]

# %%

print("Sorting entries by query length")
entries_with_idx = list(enumerate(queries))
entries_with_label = []
for index, query in entries_with_idx:
  if index in reasoning_indices:
    label = "reasoning"
  elif index in memorizing_indices:
    label = "memorizing"
  else:
    raise ValueError(f"Index {index} not found in either reasoning or memorizing indices.")
  entries_with_label.append({
    "label": label,
    "query": query,
    **test_dataset[index],
  })
entries_sorted = sorted(
  entries_with_label, 
  key=lambda x: len(x['query']),
  reverse=True,
)

del entries_with_idx
del entries_with_label

# %%

print("Preparing entries for batching")
entries_batched = [
  entries_sorted[i:i + args['batch_size']]
  for i in range(0, len(entries_sorted), args['batch_size'])
]

del entries_sorted

# %%

directions: dict[int, Float[Tensor, "n_embd"]] = {}
if args['with_intervention']:
  print("Computing directions for the model.")
  candidate_directions = t.load(
    args['candidate_directions_file_path'],
    map_location='cpu',
    weights_only=False,
  )[args['model_name']]

  for layer_index in range(len(candidate_directions['reasoning'])):
    directions[layer_index] = candidate_directions['reasoning'][layer_index] - candidate_directions['memorizing'][layer_index]

  device_map = get_device_map(model=model)

  for layer_index, direction in directions.items():
    directions[layer_index] = direction.to(
      device=device_map[layer_index],
      dtype=model.model.dtype,
    )

  del candidate_directions
else:
  print("No intervention, proceeding without directions.")

# %%

if args['with_intervention']:
  print("Setting up projection hooks for the model.")
  match model.config.model_type:
    case "huginn_raven":
      projection_hook_config = ProjectionHookConfig(
        steering_mode=args['projection_hook_mode'],
        modification_mode=args['modification_mode'],
        direction_normalization_mode=args['direction_normalization_mode'],
        layer_indices=args['layer_indices'],
        hidden_states_hooks_config={
          "pre_hook": args['with_hidden_states_pre_hook'],
          "post_hook": args['with_hidden_states_post_hook'],
        },
        scale=args['scale'],
      )
    case "llama":
      projection_hook_config = ProjectionHookConfigLiReFs(
        steering_mode=args['projection_hook_mode'],
        modification_mode=args['modification_mode'],
        direction_normalization_mode=args['direction_normalization_mode'],
        layer_indices=args['layer_indices'],
        hidden_states_hooks_config={
          "pre_hook": args['with_hidden_states_pre_hook'],
          "post_hook": args['with_hidden_states_post_hook'], # Not implemented
        },
        attention_hooks_config={
          "pre_hook": False,
          "post_hook": True,
        },
        mlp_hooks_config={
          "pre_hook": False,
          "post_hook": True,
        },
        scale=args['scale'],
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
  hooks = set_activations_hooks(
    model=model,
    feature_directions=directions,
    config=projection_hook_config,
  )
else:
  print("No intervention hooks set up, proceeding without them.")

no_answers = None

for entries_batch in tqdm(entries_batched):
  queries_batch = [item['query'] for item in entries_batch]

  inputs = tokenize_text(
    model=model,
    tokenizer=tokenizer,
    text=queries_batch,
  )

  with t.no_grad():
    outputs = generate(
      model=model,
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      tokenizer=tokenizer,
      huginn_num_steps=args.get("huginn_num_steps", None)
    )
  
  responses = tokenizer.batch_decode(
    outputs[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
  )

  no_answers = set_model_predict_correctness(
    entries=entries_batch,
    queries=queries_batch,
    responses=responses,
    test_dataset_name=args['test_data_name'],
    no_answers=no_answers,
  )

  del inputs
  del responses

  t.cuda.empty_cache()

print(f"No answers found in the dataset: {len(no_answers)}")

if args['with_intervention']:
  print("Removing projection hooks from the model.")
  for hook in hooks:
    hook.remove()
  hooks.clear()

# %%

def _compute_accuracy(
  entries: list[str],
  label: str
):
  total = len(entries)
  if total == 0:
    print(f"No entries found for {label}. Cannot compute accuracy.")
    raise None
  correct = sum(entry.get("model_predict_correctness", False) for entry in entries)
  accuracy = correct / total
  return accuracy

match args['test_data_name']:
  case 'mmlu-pro':
    reasoning_entries = [
      entry
      for batch in entries_batched
      for entry in batch
      if entry['memory_reason_score'] > 0.5
    ]
    memorizing_entries = [
      entry
      for batch in entries_batched
      for entry in batch
      if entry['memory_reason_score'] <= 0.5
    ]

    reasoning_label = f"Layer {args['layer_indices']} - Reasoning Subset"
    reasoning_accuracy = _compute_accuracy(
      entries=reasoning_entries,
      label=reasoning_label
    )
    print(f"{reasoning_label} Accuracy: {reasoning_accuracy:.4f}")
    
    memorizing_label = f"Layer {args['layer_indices']} - Memorizing Subset"
    memorizing_accuracy = _compute_accuracy(
      entries=memorizing_entries,
      label=memorizing_label
    )
    print(f"{memorizing_label} Accuracy: {memorizing_accuracy:.4f}")
  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%

if args['output_file_path'] is None:
  print("No output file path specified, skipping saving results.")
  sys.exit(0)

# %%

print(f"Loading existing results from {args['output_file_path']} if available.")
try:
  output = t.load(
    args['output_file_path'],
    map_location='cpu',
    weights_only=False,
  )
except FileNotFoundError as e:
  print(f"File not found: {args['output_file_path']}. Creating a new results dictionary.")
  output = {}

# %%

output_key = ' '.join(
  [
    f"{key}={value}"
    for key, value in args.items()
    if key in [
      'huginn_num_steps', 
      'test_data_name', 
      'with_fewshot_prompts',
  
      'with_intervention',
      'direction_normalization_mode',
      'projection_hook_mode',
      'layer_indices',
      'with_hidden_states_pre_hook',
      'with_hidden_states_post_hook',
      'scale',
    ]
  ]
)
print(f"Using output key: {output_key}")

# %%

output[output_key] = {
  'reasoning_accuracy': reasoning_accuracy,
  'memorizing_accuracy': memorizing_accuracy,
  'no_answers': no_answers,
}

output_path = os.path.dirname(args['output_file_path'])
os.makedirs(output_path, exist_ok=True)
print(f"Saving results to {args['output_file_path']}")
t.save(output, args['output_file_path'])

# %%