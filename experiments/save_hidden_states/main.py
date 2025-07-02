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
  reload(sys.modules.get('utils.prepare_queries', sys))
  reload(sys.modules.get('utils.get_n_layers', sys))
  reload(sys.modules.get('utils.get_n_embd', sys))
  reload(sys.modules.get('utils.cache_hidden_states', sys))
  reload(sys.modules.get('utils.load_hidden_states_cache', sys))
  reload(sys.modules.get('utils', sys))

from shs_utils import parse_args

from utils import use_deterministic_algorithms
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import load_and_sample_test_dataset
from utils import prepare_queries
from utils import cache_hidden_states

import torch
from torch import Tensor
from jaxtyping import Float
from tqdm import tqdm

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/media/npu-tao/disk4T/jason"
  sys.argv = [
    'main.py',
    '--models_path', f'{root_path}/transformers',
    '--model_name', 'Meta-Llama-3-8B',

    '--data_path', f'{root_path}/datasets/lirefs',
    '--data_name', 'mmlu-pro-3000samples.json',

    # '--data_sample_size', '600',
    '--data_batch_size', '8',

    '--output_file_path', f'{root_path}/experiments/hidden_states_cache/meta-llama-3-8b_mmlu-pro-3000samples.pt',
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
)

# %%

if args['data_name'].endswith('.json'):
  print(f"Loading dataset from JSON file {args['data_path']}/{args['data_name']} and sample size {args['data_sample_size']}")
  sampled_data = load_json_dataset(
    file_path=os.path.join(args['data_path'], args['data_name']),
    sample_size=args['data_sample_size'],
  )
else:
  print(f"Loading dataset {args['data_path']}/{args['data_name']} and sample size {args['data_sample_size']}")
  sampled_data = load_and_sample_test_dataset(
    data_path=args['data_path'],
    data_name=args['data_name'],
    sample_size=args['data_sample_size'],
  )

# %%

queries = prepare_queries(
  model_name=model.config.model_type,
  data=sampled_data,
  data_name=args['data_name'],
  tokenizer=tokenizer,
  apply_chat_template=False,
  system_prompt=None,
  fewshot_prompts=None,
  with_cot=False,
  with_options=False,
)

queries_batched = [
  queries[i:i + args['data_batch_size']]
  for i in range(
    0, len(queries), args['data_batch_size']
  )
]

# %%

hidden_states_cache: dict[
  int, Float[Tensor, "seq_len n_embd"]
] = None

for queries_batch in tqdm(queries_batched):
  hidden_states_cache = cache_hidden_states(
    model=model,
    tokenizer=tokenizer,
    queries_batch=queries_batch,
    hidden_states_cache=hidden_states_cache
  )
  torch.cuda.empty_cache()

# %%

output_path = os.path.dirname(args['output_file_path'])
os.makedirs(output_path, exist_ok=True)

output_file_path = args['output_file_path']
print(f"Saving hidden states cache to {output_file_path}")
torch.save(hidden_states_cache, output_file_path)

# %%
