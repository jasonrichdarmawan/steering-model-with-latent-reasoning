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

from utils import enable_reproducibility

from utils import get_device_map
from utils import ProcessHiddenStatesMode
from utils import compute_directions

from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode
from utils import ProjectionHookConfig
from utils import set_activations_hooks

from os.path import join
import torch as t
from torch import Tensor
from jaxtyping import Float
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  
  MODEL_NAME = "huginn-0125"
  
  PROCESS_HIDDEN_STATES_MODE = ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN
  
  DIRECTION_NORMALIZATION_MODE = DirectionNormalizationMode.UNIT_VECTOR
  PROJECTION_HOOK_MODE = ProjectionHookMode.FEATURE_ADDITION
  MODIFICATION_MODE = TokenModificationMode.LAST_TOKEN
  
  sys.argv = [
    'main.py',
    '--data_path', f'{WORKSPACE_PATH}/datasets',

    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    '--device', 'cuda:0',

    '--huginn_mean_recurrence', '32',

    '--with_intervention',

    '--use_linear_probes',
    '--linear_probes_file_path', f'{WORKSPACE_PATH}/experiments/train_linear_probes/best_checkpoint.pt',

    # '--use_candidate_directions',
    # '--process_hidden_states_mode', PROCESS_HIDDEN_STATES_MODE,
    # '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/{MODEL_NAME}_mmlu-pro-3000samples.json_{PROCESS_HIDDEN_STATES_MODE}_candidate_directions.pt',
    
    '--direction_normalization_mode', str(DIRECTION_NORMALIZATION_MODE),
    '--layer_indices', '31',
    '--projection_hook_mode', str(PROJECTION_HOOK_MODE),
    '--modification_mode', str(MODIFICATION_MODE),
    # '--with_hidden_states_pre_hook',
    '--with_hidden_states_post_hook',
    '--scale', '1.0',

    '--tasks', 'piqa',
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

print(f"Loading model: {args['model_name']} from path: {args['models_path']}")
match args["model_name"]:
  case "huginn-0125":
    """
    Reference: https://github.com/seal-rg/recurrent-pretraining/blob/0d9ed974d253e16498edec5c0c0916fdef4eb339/evaluate_raven/hf_eval_adaptive_compute.py
    """
    model = HFLM(
      pretrained=join(
        args["models_path"], 
        args["model_name"]
      ),
      device=args["device"],
      mean_recurrence=args["huginn_mean_recurrence"],
      dtype=t.bfloat16,
      batch_size=args["batch_size"],
    )
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

# %%

feature_directions: dict[int, Float[Tensor, "n_embd"]] = None
overall_directions_magnitude: dict[int, Float[Tensor, ""]] | None = None
if args["with_intervention"]:
  device_map = get_device_map(model=model.model)

  if args['use_linear_probes']:
    print(f"Loading linear probes from file: {args['linear_probes_file_path']}")
    checkpoint = t.load(
      args['linear_probes_file_path'],
      map_location='cpu',
      weights_only=True,
    )
    print("Best layer index:", checkpoint.get('layer_index', None))

    feature_directions = {
      layer_index: linear_probe[:, 0] - linear_probe[:, 1]
      for layer_index, linear_probe in enumerate(checkpoint['linear_probe'])
    }

    for layer_index, direction in feature_directions.items():
      feature_directions[layer_index] = direction.to(
        device=device_map[layer_index],
        dtype=model.model.dtype,
      )
    
    del checkpoint
    
  elif args['use_candidate_directions']:
    print(f"Computing directions from file: {args['candidate_directions_file_path']}")
    candidate_directions = t.load(
      args['candidate_directions_file_path'],
      map_location='cpu',
      weights_only=False,
    )

    feature_directions = compute_directions(
      model=model.model,
      candidate_directions=candidate_directions,
      positive_label="reasoning",
      negative_label="memorizing",
    )

    if args['direction_normalization_mode'] == DirectionNormalizationMode.SCALE_WITH_OVERALL_MAGNITUDE:
      print("Using overall directions magnitude for normalization.")  
      overall_directions_magnitude = {
        layer_index: candidate_direction.norm(dim=-1)
        for layer_index, candidate_direction in candidate_directions["overall"]["mean"].items()
      }

      for layer_index, direction in feature_directions.items():
        overall_directions_magnitude[layer_index] = overall_directions_magnitude[layer_index].to(
          device=device_map[layer_index],
          dtype=model.model.dtype,
        )

    del candidate_directions
  else:
    raise ValueError(
      "Either `use_linear_probes` or `use_candidate_directions` must be set to True."
    )
else:
  print("No intervention will be performed.")

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
    case _:
      raise ValueError(f"Unsupported model type for projection hooks: {model.config.model_type}")

  hooks = set_activations_hooks(
    model=model.model,
    feature_directions=feature_directions,
    config=projection_hook_config,
    overall_directions_magnitude=overall_directions_magnitude,
  )
else:
  print("No intervention hooks set up, proceeding without them.")

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
  output = t.load(
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

      'use_linear_probes',

      'use_candidate_directions',
      'process_hidden_states_mode',

      'direction_normalization_mode',
      'layer_indices',
      'projection_hook_mode',
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
t.save(output, args['output_file_path'])

# %%
