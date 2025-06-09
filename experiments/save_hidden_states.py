# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  sys.argv = [
    'extract_hidden_states.py',
    '--models_path', '/root/autodl-fs/transformers',
    '--model_name', 'huginn-0125',
    '--data_path', '/root/autodl-fs/datasets',
    '--data_name', 'mmlu-pro',
    '--data_sample_size', '600',
    '--data_batch_size', '4',
    '--output_path', '/root/autodl-fs/hidden_states_cache',
  ]

args = parse_args()

# %%

if False:
  from importlib import reload

  print("Reloading modules to ensure the latest code is used.")
  reload(sys.modules.get('utils.use_deterministic_algorithms', sys))
  reload(sys.modules.get('utils.parse_args', sys))
  reload(sys.modules.get('utils.load_model_and_tokenizer', sys))
  reload(sys.modules.get('utils.load_and_sample_test_dataset', sys))
  reload(sys.modules.get('utils.cache_hidden_states', sys))
  reload(sys.modules.get('utils', sys))

from utils import use_deterministic_algorithms
from utils import parse_args
from utils import load_model_and_tokenizer
from utils import load_and_sample_test_dataset
from utils import cache_hidden_states

import os
import torch

# %%

print("Setting deterministic algorithms for reproducibility.")
use_deterministic_algorithms()

# %%

model, tokenizer = load_model_and_tokenizer(
  model_path=args['models_path'],
  model_name=args['model_name'],
)

# %%

sampled_data = load_and_sample_test_dataset(
  data_path=args['data_path'],
  data_name=args['data_name'],
  sample_size=args['data_sample_size']
)

# %%

hidden_states_cache = cache_hidden_states(
  data=sampled_data,
  model=model,
  tokenizer=tokenizer,
  data_batch_size=args['data_batch_size'],
)

# %%

os.makedirs(args['output_path'], exist_ok=True)

torch.save(
  hidden_states_cache, 
  os.path.join(args['output_path'], f'{args["model_name"]}_hidden_states.pt')
)

# %%
