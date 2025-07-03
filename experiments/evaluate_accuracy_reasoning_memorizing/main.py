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
from utils import load_hidden_states_cache
from utils import load_json_dataset
from utils import compute_candidate_directions
from utils import prepare_fewshot_prompts
from utils import prepare_queries
from utils import ProjectionHookConfig
from utils import ProjectionHookConfigLiReFs
from utils import set_activations_hooks
from utils import generate_sentences_huginn
from utils import generate_sentences_lirefs
from utils import set_model_predict_correctness

from tqdm import tqdm
import torch

# %%

if True:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/media/npu-tao/disk4T/jason"
  sys.argv = [
    'main.py',
    '--models_path', f'{root_path}/transformers',
    '--model_name', 'Meta-Llama-3-8B',

    # '--huginn_num_steps', '32',

    '--test_data_path', f'{root_path}/datasets/lirefs',
    '--test_data_name', 'mmlu-pro',
    '--test_data_sample_size', '200',
    '--with_fewshot_prompts',
    # '--with_cot',
    '--batch_size', '1',

    '--with_intervention',
    '--hidden_states_data_file_path', f'{root_path}/datasets/lirefs/mmlu-pro-3000samples.json',
    '--hidden_states_cache_file_path', f'{root_path}/experiments/hidden_states_cache/Meta-Llama-3-8B_mmlu-pro-3000samples.pt',
    '--layer_indices', '21',
    '--with_hidden_states_pre_hook',
    # '--with_hidden_states_post_hook',
    '--scale', '0.1',

    '--output_file_path', f'{root_path}/experiments/reasoning_memorizing_accuracy/Meta-Llama-3-8B.json'
  ]

args = parse_args()
print(f"Parsed arguments:")
print('#' * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Setting deterministic algorithms for reproducibility.")
enable_reproducibility()

if args['batch_size'] > 1:
  print("torch.nn.functional.scaled_dot_product_attention (e.g., used in Meta-Llama-3-8B) throws error when batch size is larger than 1 and the package models.recpre is imported")

# %%

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
    device_map = "auto"
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
  device_map=device_map,
)

# %%

# Load the test dataset based on the specified test data name.
# TODO: Isn't this data leakage?
# The mmlu-pro-3000samples and the mmlu-pro datasets
# are used for both extracting candidate directions and testing the model.
# Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/main/reasoning_representation/Intervention/features_intervention.py
match args['test_data_name']:
  case "mmlu-pro":
    test_dataset = load_json_dataset(
      file_path=args["hidden_states_data_file_path"],
      sample_size=args['test_data_sample_size'],
    )
    print(f"Loaded test dataset from {args['hidden_states_data_file_path']} with {len(test_dataset)} samples.")
  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%

if args['with_fewshot_prompts']:
  fewshot_prompts = prepare_fewshot_prompts(
    data_path=args['test_data_path'],
    data_name=args['test_data_name'],
    with_cot=args['with_cot'],
  )
  print(f"Prepared few-shot prompts for {args['test_data_name']} with {len(fewshot_prompts)} samples.")
else:
  print("No few-shot prompts will be used for the test dataset.")
  fewshot_prompts = None

# %%

queries = prepare_queries(
  model_name=model.config.model_type,
  data=test_dataset,
  data_name=args['test_data_name'],
  tokenizer=tokenizer,
  apply_chat_template=False,
  fewshot_prompts=fewshot_prompts,
  with_cot=args['with_cot'],
)

queries_batched = [
  queries[i:i + args['batch_size']]
  for i in range(
    0,
    len(queries),
    args['batch_size']
  )
]

entries_batched = [
  test_dataset[i:i + args['batch_size']]
  for i in range(
    0,
    len(test_dataset),
    args['batch_size']
  )
]

# %%

if args['with_intervention']:
  hidden_states_data_file_path = load_json_dataset(
    file_path=args['hidden_states_data_file_path'],
  )
  print(f"Loaded hidden states dataset from {args['hidden_states_data_file_path']} with {len(hidden_states_data_file_path)} samples.")

  reasoning_indices = [
    index for index, sample in enumerate(hidden_states_data_file_path) 
    if sample['memory_reason_score'] > 0.5
  ]
  memorizing_indices = [
    index for index, sample in enumerate(hidden_states_data_file_path) 
    if sample['memory_reason_score'] <= 0.5
  ]

# %%

if args['with_intervention']:
  hidden_states_cache = load_hidden_states_cache(
    file_path=args['hidden_states_cache_file_path'],
  )
  print(f"Loaded hidden states cache from {args['hidden_states_cache_file_path']} with {len(hidden_states_cache)} layers.")

# %%

if args['with_intervention']:
  candidate_directions = compute_candidate_directions(
    model=model,
    hidden_states_cache=hidden_states_cache,
    reasoning_indices=reasoning_indices,
    memorizing_indices=memorizing_indices,
    layer_indices=args['layer_indices'],
  )
  print(f"Computed candidate directions for layers {args['layer_indices']} with {len(candidate_directions)} candidate directions.")

# %%

if args['with_intervention'] is False:
  print("No intervention will be performed, skipping hidden states cache and candidate directions computation.")
  candidate_directions = None

# %%

if args['with_intervention']:
  print("Setting up projection hooks for the model.")
  match model.config.model_type:
    case "huginn_raven":
      projection_hook_config = ProjectionHookConfig(
        layer_indices=args['layer_indices'],
        candidate_directions=candidate_directions,
        hidden_states_hooks={
          "pre_hook": args['with_hidden_states_pre_hook'],
          "post_hook": args['with_hidden_states_post_hook'],
        },
        scale=args['scale'],
      )
    case "llama":
      projection_hook_config = ProjectionHookConfigLiReFs(
        layer_indices=args['layer_indices'],
        candidate_directions=candidate_directions,
        hidden_states_hooks={
          "pre_hook": args['with_hidden_states_pre_hook'],
          "post_hook": args['with_hidden_states_post_hook'], # Not implemented
        },
        attention_hooks={
          "pre_hook": False, # Not implemented
          "post_hook": True,
        },
        mlp_hooks={
          "pre_hook": False, # Not implemented
          "post_hook": True,
        },
        scale=args['scale'],
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
  hooks = set_activations_hooks(
    model=model,
    candidate_directions=candidate_directions,
    config=projection_hook_config,
  )
else:
  print("No intervention hooks set up, proceeding without them.")

# %%

no_answers = None

for queries_batch, entries_batch in tqdm(
  zip(queries_batched, entries_batched),
  total=len(queries_batched)
):
  match model.config.model_type:
    case "huginn_raven":
      responses = generate_sentences_huginn(
        model=model,
        tokenizer=tokenizer,
        text=queries_batch,
        num_steps=args["huginn_num_steps"],
      )
    case "llama":
      responses = generate_sentences_lirefs(
        model=model,
        tokenizer=tokenizer,
        text=queries_batch,
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")

  no_answers = set_model_predict_correctness(
    entries=entries_batch,
    queries=queries_batch,
    responses=responses,
    test_dataset_name=args['test_data_name'],
    no_answers=no_answers,
  )

  torch.cuda.empty_cache()

# %%

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
  print(f"{label} Accuracy: {accuracy:.4f} ({correct}/{total})")
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

    reasoning_accuracy = _compute_accuracy(
      entries=reasoning_entries,
      label=f"Layer {args['layer_indices']} - Reasoning Subset"
    )
    memorizing_accuracy = _compute_accuracy(
      entries=memorizing_entries,
      label=f"Layer {args['layer_indices']} - Memorizing Subset"
    )
  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%

if args['output_file_path'] is None:
  print("No output file path specified, skipping saving results.")
  sys.exit(0)

# %%

try:
  output = torch.load(
    args['output_file_path'],
    map_location='cpu',
    weights_only=True
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
      'layer_indices',
      'with_hidden_states_pre_hook',
      'with_hidden_states_post_hook',
      'scale'
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
torch.save(output, args['output_file_path'])

# %%