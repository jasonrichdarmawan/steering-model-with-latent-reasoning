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

from ele_utils import parse_args

from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import ProcessHiddenStatesMode
from utils import enable_reproducibility
from utils import load_model
from utils import HuginnWrapper
from utils import compute_directions
from utils import ProjectionHookConfig
from utils import set_activations_hooks

from os.path import join
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  MODEL_NAME = "huginn-0125"
  PROCESS_HIDDEN_STATES_MODE = str(ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN)
  DIRECTION_NORMALIZATION_MODE = str(DirectionNormalizationMode.UNIT_VECTOR)
  PROJECTION_HOOK_MODE = str(ProjectionHookMode.FEATURE_ADDITION)
  sys.argv = [
    'main.py',
    '--data_path', f'{WORKSPACE_PATH}/datasets',

    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    '--huginn_mean_recurrence', '32',

    '--with_intervention',
    '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/{MODEL_NAME}_mmlu-pro-3000samples.json_{PROCESS_HIDDEN_STATES_MODE}_candidate_directions.pt',
    '--direction_normalization_mode', DIRECTION_NORMALIZATION_MODE,
    '--projection_hook_mode', PROJECTION_HOOK_MODE,
    '--layer_indices', '129',
    # '--with_hidden_states_pre_hook',
    '--with_hidden_states_post_hook',
    '--scale', '1.0',

    '--tasks', 'mmlu',
    '--num_fewshot', '0',
    '--batch_size', '1',
    '--limit', '1',
    '--output_file_path', f'{WORKSPACE_PATH}/experiments/lm_eval_results/{MODEL_NAME}.json',
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
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

# %%

match args["model_name"]:
  case "huginn-0125":
    """
    # Reference: https://github.com/seal-rg/recurrent-pretraining/blob/0d9ed974d253e16498edec5c0c0916fdef4eb339/evaluate_raven/hf_eval_adaptive_compute.py
    """
    # model = load_model(
    #   models_path=args["models_path"],
    #   model_name=args["model_name"],
    #   device_map=device_map,
    # )
    model = HFLM(
      pretrained=join(
        args["models_path"], 
        args["model_name"]
      ),
      parallelize=True,
      device_map=device_map,
      mean_recurrence=args["huginn_mean_recurrence"],
      dtype=torch.bfloat16,
    )

    # model = HuginnWrapper(
    #   pretrained=join(
    #     args["models_path"], 
    #     args["model_name"]
    #   ),
    #   backend="causal",
    #   device_map=device_map,
    #   batch_size=args["batch_size"],
    #   trust_remote_code=False,
    #   dtype=torch.bfloat16,
    #   criterion="entropy-diff",
    #   exit_threshold="auto",
    #   lookup_strategy="full",
    #   continuous_compute=False,
    #   latent_dampening=False,
    # )
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

# %%

if args["with_intervention"]:
  print(f"Computing directions from file: {args['candidate_directions_file_path']}")
  candidate_directions = torch.load(
    args['candidate_directions_file_path'],
    map_location='cpu',
    weights_only=False,
  )

  directions = compute_directions(
    model=model.model,
    candidate_directions=candidate_directions,
    positive_label="reasoning",
    negative_label="memorizing",
    overall_label="overall",
    normalization_mode=args['direction_normalization_mode'],
  )

  del candidate_directions
else:
  print("No intervention will be performed, skipping hidden states cache and candidate directions computation.")
  candidate_directions = None

# %%

if args['with_intervention']:
  print("Setting up projection hooks for the model.")
  match model.config.model_type:
    case "huginn_raven":
      projection_hook_config = ProjectionHookConfig(
        mode=args['projection_hook_mode'],
        layer_indices=args['layer_indices'],
        directions=directions,
        hidden_states_hooks={
          "pre_hook": args['with_hidden_states_pre_hook'],
          "post_hook": args['with_hidden_states_post_hook'],
        },
        scale=args['scale'],
      )
    case _:
      raise ValueError(f"Unsupported model type for projection hooks: {model.config.model_type}")

  hooks = set_activations_hooks(
    model=model.model,
    directions=directions,
    config=projection_hook_config,
  )
else:
  print("No intervention hooks set up, proceeding without them.")

# %%

from typing import Optional, Any
import datasets
from lm_eval.api.task import ConfigurableTask

def download(self, dataset_kwargs: Optional[dict[str, Any]] = None) -> None:
  path = os.path.join(
    args["data_path"], 
    self.DATASET_PATH,
  )
  print(f"Loading dataset from local path: {path}")
  self.dataset = datasets.load_dataset(
    path=path,
    name=self.DATASET_NAME,
    **dataset_kwargs if dataset_kwargs is not None else {},
  )

print(
  "Overriding ConfigurableTask.download to load datasets from local path.\n"
  f"Specified location: {args['data_path']}/<dataset-name>\n"
  "Ensure dataset files (including Python scripts) exist at the specified location.\n"
)
ConfigurableTask.download = download

# %%

match args["model_name"]:
  case "huginn-0125":
    results = simple_evaluate(
      model=model,
      tasks=args['tasks'],
      num_fewshot=args["num_fewshot"],
      limit=args["limit"],
    )["results"]
    print(f"Evaluation results:\n{results}")
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

# %%

if args["with_intervention"]:
  print("Removing projection hooks from the model.")
  for hook in hooks:
    hook.remove()

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
      
      'huginn_mean_recurrence',
      
      'with_intervention',
      'direction_normalization_mode',
      'projection_hook_mode',
      'layer_indices',
      'with_hidden_states_pre_hook',
      'with_hidden_states_post_hook',
      'scale',

      'tasks',
      'num_fewshot',
      'limit',
    ]
  ]
)
print(f"Using output key: {output_key}")

# %%

output[output_key] = results

# %%

output_path = os.path.dirname(args['output_file_path'])
os.makedirs(output_path, exist_ok=True)

print(f"Saving evaluation results to: {args['output_file_path']}")
torch.save(output, args['output_file_path'])

# %%
