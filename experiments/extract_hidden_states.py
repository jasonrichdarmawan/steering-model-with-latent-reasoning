# %%

import sys

if False:
  print("Programatically setting sys.argv for testing purposes.")
  sys.argv = [
    'extract_hidden_states.py',
    '--models_path', '/root/autodl-fs/transformers',
    '--model_name', 'huginn-0125',
    '--data_path', '/root/autodl-fs/datasets',
    '--data_name', 'mmlu-pro',
  ]

# %%

from utils import use_deterministic_algorithms
from utils import parse_args
from utils import load_model_and_tokenizer

from datasets import load_dataset
from os.path import join

use_deterministic_algorithms()

args = parse_args()

# %%

model, tokenizer = load_model_and_tokenizer(
  model_path=args['models_path'],
  model_name=args['model_name'],
)

# %%

dataset = load_dataset(
  path=join(args['data_path'], 
            args['data_name']),
)

# %%
