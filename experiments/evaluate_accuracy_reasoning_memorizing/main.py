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
  
  reload(sys.modules.get('earm_utils.parse_args', sys))
  reload(sys.modules.get('earm_utils', sys))

  reload(sys.modules.get('utils.use_deterministic_algorithms', sys))
  reload(sys.modules.get('utils.load_model_and_tokenizer', sys))
  reload(sys.modules.get('utils.load_hidden_states_cache', sys))
  reload(sys.modules.get('utils.load_json_dataset', sys))
  reload(sys.modules.get('utils.compute_candidate_directions', sys))
  reload(sys.modules.get('utils.prepare_fewshot_prompts', sys))
  reload(sys.modules.get('utils.prepare_queries', sys))
  reload(sys.modules.get('utils.set_activations_hooks', sys))
  reload(sys.modules.get('utils.remove_hooks', sys))
  reload(sys.modules.get('utils.generate_sentences_huginn', sys))
  reload(sys.modules.get('utils', sys))

from earm_utils import parse_args

from utils import use_deterministic_algorithms
from utils import load_model_and_tokenizer
from utils import load_hidden_states_cache
from utils import load_json_dataset
from utils import compute_candidate_directions
from utils import prepare_fewshot_prompts
from utils import prepare_queries
from utils import generate_sentences_huginn
from utils import ProjectionHookConfig
from utils import set_activations_hooks, remove_hooks
from utils import set_model_predict_correctness

from tqdm import tqdm
import torch

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/home/npu-tao/jason"
  sys.argv = [
    'main.py',
    '--models_path', f'{root_path}/transformers',
    '--model_name', 'huginn-0125',
    '--device', 'auto',

    '--huginn_num_steps', '32',

    '--test_data_path', f'{root_path}/datasets/lirefs',
    '--test_data_name', 'mmlu-pro-3000samples.json',
    '--with_fewshot_prompts',
    # '--with_cot',
    '--batch_size', '1',

    '--with_intervention',
    '--hidden_states_cache_file_path', f'{root_path}/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt',
    '--layer_indices', '66',
    # '--with_pre_hook',
    '--with_post_hook',
    '--scale', '0.1',

    '--output_file_path', f'{root_path}/experiments/reasoning_memorizing_accuracy/huginn-0125.json'
  ]

args = parse_args()
print(f"Parsed arguments:")
print('#' * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Setting deterministic algorithms for reproducibility.")
use_deterministic_algorithms()

# %%

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
  device=args["device"],
)

# %%

if args['with_intervention']:
  print("Loading mmlu-pro-3000samples dataset for intervention.")
  file_path = os.path.join(
    args['test_data_path'],
    "mmlu-pro-3000samples.json"
  )
  mmlu_pro_3000samples_dataset = load_json_dataset(
    file_path=file_path,
  )

  reasoning_indices = [
    index for index, sample in enumerate(mmlu_pro_3000samples_dataset) 
    if sample['memory_reason_score'] > 0.5
  ]
  memorizing_indices = [
    index for index, sample in enumerate(mmlu_pro_3000samples_dataset) 
    if sample['memory_reason_score'] <= 0.5
  ]

# %%

if args['with_intervention']:
  print("Loading hidden states cache for intervention.")
  hidden_states_cache = load_hidden_states_cache(
    file_path=args['hidden_states_cache_file_path'],
  )

  candidate_directions = compute_candidate_directions(
    hidden_states_cache=hidden_states_cache,
    reasoning_indices=reasoning_indices,
    memorizing_indices=memorizing_indices,
    dtype=model.dtype
  )
else:
  print("No intervention will be performed, skipping hidden states cache and candidate directions computation.")
  candidate_directions = None

# %%

# Load the test dataset based on the specified test data name.
# TODO: Isn't this data leakage?
# The mmlu-pro-3000samples dataset is used for both extracting candidate directions and testing the model.
# Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/main/reasoning_representation/Intervention/features_intervention.py
match args['test_data_name']:
  case 'mmlu-pro-3000samples.json':
    file_path = os.path.join(
      args['test_data_path'], 
      "mmlu-pro-3000samples.json"
    )
    print(f"Loading {file_path} dataset with sample_size {args['test_data_sample_size']} for testing.")
    test_dataset = load_json_dataset(
      file_path=file_path,
      sample_size=args['test_data_sample_size'],
    )

    reasoning_indices = [index for index, sample in enumerate(test_dataset) if sample['memory_reason_score'] > 0.5]
    memorizing_indices = [index for index, sample in enumerate(test_dataset) if sample['memory_reason_score'] <= 0.5]

  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%

if args['with_fewshot_prompts']:
  print("Preparing few-shot prompts for the test dataset.")
  fewshot_prompts = prepare_fewshot_prompts(
    data_path=args['test_data_path'],
    data_name=args['test_data_name'],
    with_cot=args['with_cot'],
  )
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

if args['with_intervention']:
  print("Setting up projection hooks for the model.")
  projection_hook_config = ProjectionHookConfig(
    layer_indices=args['layer_indices'],
    candidate_directions=candidate_directions,
    pre_hook=args['with_pre_hook'],
    post_hook=args['with_post_hook'],
    scale=args['scale']
  )

  hooks = set_activations_hooks(
    model=model,
    candidate_directions=candidate_directions,
    config=projection_hook_config,
  )
else:
  print("No intervention hooks set up, proceeding without them.")

for queries_batch, entries_batch in tqdm(
  zip(queries_batched, entries_batched),
  total=len(queries_batched)
):
  match model.config.model_type:
    case name if "huginn_" in name:
      responses = generate_sentences_huginn(
        model=model,
        tokenizer=tokenizer,
        text=queries_batch,
        num_steps=args["huginn_num_steps"],
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")

  set_model_predict_correctness(
    entries=entries_batch,
    queries=queries_batch,
    responses=responses,
    test_dataset_name=args['test_data_name']
  )

  torch.cuda.empty_cache()
    
if args['with_intervention']:
  remove_hooks(hooks)

# %%

match args['test_data_name']:
  case 'mmlu-pro-3000samples.json':
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
      'with_pre_hook',
      'with_post_hook',
      'scale'
    ]
  ]
)
print(f"Using output key: {output_key}")

# %%

output[output_key] = {
  'reasoning_accuracy': reasoning_accuracy,
  'memorizing_accuracy': memorizing_accuracy,
}

output_path = os.path.dirname(args['output_file_path'])
os.makedirs(output_path, exist_ok=True)
print(f"Saving results to {args['output_file_path']}")
torch.save(output, args['output_file_path'])

# %%