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
  from utils import reload_modules

  reload_modules(
    project_root=project_root,
  )

from shs_utils import parse_args

from utils import enable_reproducibility
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import prepare_queries
from utils import get_n_layers
from utils import tokenize_text
from utils import get_hidden_states
from utils import CacheHiddenStatesMode
from utils import SaveHiddenStatesQueryLabel
from utils import SaveHiddenStatesOutput

from tqdm import tqdm
import torch as t
from typing import TypedDict

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  sys.argv = [
    'main.py',
    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', 'huginn-0125',

    '--data_path', f'{WORKSPACE_PATH}/datasets/lirefs',
    '--data_name', 'mmlu-pro-3000samples.json',
    # '--data_sample_size', '24',
    '--data_batch_size', '8',

    '--cache_hidden_states_mode', str(CacheHiddenStatesMode.FIRST_ANSWER_TOKEN),

    '--output_path', f'{WORKSPACE_PATH}/experiments/save_hidden_states',
  ]

args = parse_args()
print("Parsed arguments:")
print("#" * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Setting deterministic behavior for reproducibility.")
enable_reproducibility()

# %%

print("Loading model and tokenizer.")
model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
)

# %%

if args['data_name'] == 'mmlu-pro-3000samples.json':
  print(f"Loading dataset from JSON file {args['data_path']}/{args['data_name']} and sample size {args['data_sample_size']}")
  sampled_data = load_json_dataset(
    file_path=os.path.join(args['data_path'], args['data_name']),
    sample_size=args['data_sample_size'],
  )
else:
  raise ValueError(f"Unsupported data name: {args['data_name']}.")

# %%

print(f"Splitting the sampled data indices into reasoning and memorizing indices")
reasoning_indices = [
  index for index, sample in enumerate(sampled_data) 
  if sample['memory_reason_score'] > 0.5
]
memorizing_indices = [
  index for index, sample in enumerate(sampled_data) 
  if sample['memory_reason_score'] <= 0.5
]

# %%

print("Preparing queries for the model")
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

del sampled_data

# %%

class QueryWithLabel(TypedDict):
  label: SaveHiddenStatesQueryLabel
  query: str

print("Sorting queries by query length")
queries_with_idx = list(enumerate(queries))
queries_with_label: list[QueryWithLabel] = []
for index, query in queries_with_idx:
  if index in reasoning_indices:
    label = SaveHiddenStatesQueryLabel.REASONING
  elif index in memorizing_indices:
    label = SaveHiddenStatesQueryLabel.MEMORIZING
  else:
    raise ValueError(
      f"Index {index} not found in either reasoning or memorizing indices."
    )
  queries_with_label.append({
    'label': label,
    'query': query,
  })
queries_sorted = sorted(queries_with_label, key=lambda x: len(x['query']), reverse=True)

del queries_with_idx
del queries_with_label

# %%

queries_batched = [
  queries_sorted[i:i + args['data_batch_size']]
  for i in range(0, len(queries_sorted), args['data_batch_size'])
]

del queries_sorted

# %%

output: SaveHiddenStatesOutput = {
  'queries': [],
  'query_token_lengths': [],
  'labels': [],
  'hidden_states': {},
}

n_layers_to_cache = range(get_n_layers(model))

for queries_batch in tqdm(queries_batched):
  inputs = tokenize_text(
    model=model,
    tokenizer=tokenizer,
    text=[x['query'] for x in queries_batch]
  )

  match args['cache_hidden_states_mode']:
    case CacheHiddenStatesMode.FIRST_ANSWER_TOKEN:
      with t.no_grad():
        hidden_states = get_hidden_states(
          model=model,
          input_ids=inputs['input_ids'],
          attention_mask=inputs['attention_mask'],
        )
    case _:
      raise ValueError(
        f"Unsupported cache hidden states mode: {args['cache_hidden_states_mode']}"
      )
  
  for i, item in enumerate(queries_batch):
    output['queries'].append(item['query'])
    output['labels'].append(str(item['label']))
    for layer_index in n_layers_to_cache:
      if layer_index not in output['hidden_states']:
        output['hidden_states'][layer_index] = []
      
      match args['cache_hidden_states_mode']:
        case CacheHiddenStatesMode.FIRST_ANSWER_TOKEN:
          output['hidden_states'][layer_index].append(
            hidden_states[layer_index][i, -1].cpu().clone()
          )
        case _:
          raise ValueError(
            f"Unsupported cache hidden states mode: {args['cache_hidden_states_mode']}"
          )

# %%

os.makedirs(args['output_path'], exist_ok=True)
output_file_path = os.path.join(
  args['output_path'], 
  f"{args['model_name']}_hidden_states_{args['cache_hidden_states_mode']}.pt"
)
print(f"Saving hidden states to {output_file_path}")
t.save(output, output_file_path)

# %%
