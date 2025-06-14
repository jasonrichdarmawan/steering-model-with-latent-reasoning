# %%

import os
import sys

# To be able to import modules from the shs_utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")
  sys.path.insert(0, project_root)

# %%

if False:
  import sys
  from importlib import reload

  print("Reloading modules to ensure the latest code is used.")
  
  reload(sys.modules.get('shs_utils.parse_args', sys))
  reload(sys.modules.get('shs_utils', sys))

  reload(sys.modules.get('utils.use_deterministic_algorithms', sys))
  reload(sys.modules.get('utils.load_model_and_tokenizer', sys))
  reload(sys.modules.get('utils.load_and_sample_test_dataset', sys))
  reload(sys.modules.get('utils.load_json_dataset', sys))
  reload(sys.modules.get('utils.cache_hidden_states', sys))
  reload(sys.modules.get('utils.load_hidden_states_cache', sys))
  reload(sys.modules.get('utils', sys))

from shs_utils import parse_args

from utils import use_deterministic_algorithms
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import load_and_sample_test_dataset
from utils import cache_hidden_states
from utils import load_hidden_states_cache

import torch

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  sys.argv = [
    'extract_hidden_states.py',
    '--models_path', '/root/autodl-fs/transformers',
    '--model_name', 'huginn-0125',

    '--data_file_path', '/root/autodl-fs/datasets/mmlu-pro-3000samples.json',
    # '--data_path', '/root/autodl-fs/datasets',
    '--data_name', 'mmlu-pro-3000samples',

    # '--data_sample_size', '600',
    '--data_batch_size', '8',

    '--output_path', '/root/autodl-fs/experiments/hidden_states_cache',
  ]

args = parse_args()

# %%

print("Setting deterministic algorithms for reproducibility.")
use_deterministic_algorithms()

# %%

model, tokenizer = load_model_and_tokenizer(
  model_path=args['models_path'],
  model_name=args['model_name'],
)

# %%

if args['data_file_path']:
  print(f"Loading dataset from file: {args['data_file_path']}")
  sampled_data = load_json_dataset(
    file_path=args['data_file_path'],
    sample_size=args.get('data_sample_size', None)
  )
elif args['data_path'] and args['data_name']:
  print(f"Loading dataset from path: {args['data_path']}, name: {args['data_name']}")
  sampled_data = load_and_sample_test_dataset(
    data_path=args['data_path'],
    data_name=args['data_name'],
    sample_size=args.get('data_sample_size', None)
  )
else:
  raise ValueError("Either data_file_path must be provided or both data_path and data_name must be specified.")

# %%

hidden_states_cache = cache_hidden_states(
  data=sampled_data,
  model=model,
  tokenizer=tokenizer,
  data_batch_size=args['data_batch_size'],
)

# %%

os.makedirs(args['output_path'], exist_ok=True)

# %%

output = load_hidden_states_cache(
  hidden_states_cache_path=args['output_path'],
  model_name=args['model_name']
)

# %%

output[args['data_name']] = hidden_states_cache

# %%

output_file_path = os.path.join(args['output_path'], f"{args['model_name']}_hidden_states_cache.pt")

print(f"Saving hidden states cache to {output_file_path}")
torch.save(output, output_file_path)

# %%
