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

  reload(sys.modules.get('ele_utils.parse_args', sys))
  reload(sys.modules.get('ele_utils', sys))

  reload(sys.modules.get('utils.use_deterministic_algorithms', sys))
  reload(sys.modules.get('utils.load_model_and_tokenizer', sys))
  reload(sys.modules.get('utils.huginn_wrapper', sys))
  reload(sys.modules.get('utils', sys))

from ele_utils import parse_args

from utils import use_deterministic_algorithms
from utils import HuginnWrapper

from os.path import join
import torch
from lm_eval import simple_evaluate

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/home/npu-tao/jason"
  sys.argv = [
    'main.py',
    '--use_local_datasets',
    '--data_path', f'{root_path}/datasets',

    '--models_path', f'{root_path}/transformers',
    '--model_name', 'huginn-0125',
    '--huginn_model_criterion', 'entropy-diff',

    '--tasks', 'mmlu',
    '--num_fewshot', '5',
    '--batch_size', '4',
    '--limit', '50',
    '--huginn_num_steps', '32',
    '--output_path', f'{root_path}/experiments/lm_eval_results',
  ]

args = parse_args()

# %%

print("Setting deterministic algorithms for reproducibility.")
use_deterministic_algorithms()

# %%

match args["model_name"]:
  case name if name.startswith("huginn-"):
    """
    # Reference: https://github.com/seal-rg/recurrent-pretraining/blob/0d9ed974d253e16498edec5c0c0916fdef4eb339/evaluate_raven/hf_eval_adaptive_compute.py
    """
    model = HuginnWrapper(
      pretrained=join(
        args["models_path"], 
        args["model_name"]
      ),
      device="cuda:1",
      batch_size=args["batch_size"],
      trust_remote_code=False,
      dtype=torch.bfloat16,
      criterion=args["huginn_model_criterion"],
      exit_threshold="auto",
      lookup_strategy="full",
      continuous_compute=False,
      latent_dampening=False,
    )
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

# %%

if args["use_local_datasets"]:
  from typing import Optional, Any
  import datasets
  from lm_eval.api.task import ConfigurableTask

  def download(self, dataset_kwargs: Optional[dict[str, Any]] = None) -> None:
    path = os.path.join(
      args["data_path"], 
      self.DATASET_PATH,
    )
    print(f"Loading dataset {self.DATASET_NAME} from local path: {path}")
    self.dataset = datasets.load_dataset(
      path=path,
      name=self.DATASET_NAME,
      **dataset_kwargs if dataset_kwargs is not None else {},
    )

  print(
    "Overriding ConfigurableTask.download to load datasets from local path. "
    "Ensure dataset files (including Python scripts) exist at the specified location. "
    "If using a mirror, update dataset scripts to use hf-mirror.com if huggingface.co is inaccessible."
  )
  ConfigurableTask.download = download

# %%

match args["model_name"]:
  case name if name.startswith("huginn-"):
    results = simple_evaluate(
      model=model,
      tasks=args['tasks'],
      num_fewshot=args["num_fewshot"],
      limit=args["limit"],
      gen_kwargs=f"num_steps={args['huginn_num_steps']}",
    )
  case _:
    raise ValueError(f"Unsupported model name: {args['model_name']}")

# %%

if args['output_path'] is None:
  print("No output path specified. Results will not be saved.")
  sys.exit(0)
  
output_file_path = join(
  args['output_path'], 
  f"{args['model_name']}_lm_eval_results.json"
)
try:
  output = torch.load(
    output_file_path, 
    map_location='cpu',
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
      'huginn_model_criterion', 
      'tasks', 
      'num_fewshot', 
      'huginn_num_steps',
    ]
  ]
)
print(f"Using output key: {output_key}")
output[output_key] = results

# %%

os.makedirs(args['output_path'], exist_ok=True)

print(f"Saving evaluation results to: {output_file_path}")
torch.save(output, output_file_path)

# %%
