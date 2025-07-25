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
from utils import compute_directions

from utils import prepare_fewshot_prompts
from utils import prepare_queries

from utils import ProcessHiddenStatesMode

from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode
from utils import ProjectionHookConfig
from utils import ProjectionHookConfigLiReFs
from utils import set_activations_hooks

from utils import tokenize_text
from utils import generate
from utils import set_model_predict_correctness

from tqdm import tqdm
import torch

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  MODEL_NAME="huginn-0125"
  PROCESS_HIDDEN_STATES_MODE = ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN
  DIRECTION_NORMALIZATION_MODE = DirectionNormalizationMode.UNIT_VECTOR
  PROJECTION_HOOK_MODE = ProjectionHookMode.FEATURE_ADDITION
  MODIFICATION_MODE = TokenModificationMode.ALL_TOKENS
  sys.argv = [
    'main.py',
    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    '--huginn_num_steps', '129',

    '--test_data_path', f'{WORKSPACE_PATH}/datasets/lirefs',
    '--test_data_name', 'mmlu-pro',
    '--test_data_sample_size', '24',
    '--with_fewshot_prompts',
    # '--with_cot',
    '--batch_size', '1',

    '--with_intervention',
    
    '--process_hidden_states_mode', str(PROCESS_HIDDEN_STATES_MODE),
    '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/{MODEL_NAME}_mmlu-pro-3000samples.json_{PROCESS_HIDDEN_STATES_MODE}_candidate_directions.pt',
    
    '--layer_indices', '21',
    '--direction_normalization_mode', str(DIRECTION_NORMALIZATION_MODE),
    '--projection_hook_mode', str(PROJECTION_HOOK_MODE),
    '--modification_mode', str(MODIFICATION_MODE),
    '--with_hidden_states_pre_hook',
    # '--with_hidden_states_post_hook',
    '--scale', '1.0',

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

feature_directions = None
overall_directions_magnitude = None
if args['with_intervention']:
  print("Computing directions for the model.")
  candidate_directions = torch.load(
    args['candidate_directions_file_path'],
    map_location='cpu',
    weights_only=False,
  )
  feature_directions = compute_directions(
    model=model,
    candidate_directions=candidate_directions,
    positive_label="reasoning",
    negative_label="memorizing",
  )

  if args['direction_normalization_mode'] == DirectionNormalizationMode.SCALE_WITH_OVERALL_MAGNITUDE:
    overall_directions_magnitude = {
      layer_index: candidate_direction.norm(dim=-1)
      for layer_index, candidate_direction in candidate_directions["overall"]["mean"].items()
    }
    print("Using overall directions magnitude for normalization.")

  del candidate_directions

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
        direction_normalization_mode=args['direction_normalization_mode'],
        layer_indices=args['layer_indices'],
        hidden_states_hooks_config={
          "pre_hook": args['with_hidden_states_pre_hook'],
          "post_hook": args['with_hidden_states_post_hook'], # Not implemented
        },
        attention_hooks_config={
          "pre_hook": False, # Not implemented
          "post_hook": True,
        },
        mlp_hooks_config={
          "pre_hook": False, # Not implemented
          "post_hook": True,
        },
        scale=args['scale'],
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
  hooks = set_activations_hooks(
    model=model,
    feature_directions=feature_directions,
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
  inputs = tokenize_text(
    model=model,
    tokenizer=tokenizer,
    text=queries_batch,
  )

  with torch.no_grad():
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

  torch.cuda.empty_cache()

# %%

print(f"No answers found in the dataset: {len(no_answers)}")

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
  output = torch.load(
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
      'process_hidden_states_mode',
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
torch.save(output, args['output_file_path'])

# %%