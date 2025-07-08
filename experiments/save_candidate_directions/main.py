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

from scd_utils import parse_args

from utils import enable_reproducibility
from utils import load_model_and_tokenizer
from utils import load_json_dataset
from utils import prepare_queries
from utils import CandidateDirectionStats
from utils import ProcessHiddenStatesMode
from utils import tokenize_text
from utils import get_hidden_states
from utils import process_hidden_states
from utils import generate
from utils import convert_defaultdict_to_dict
from utils import compute_candidate_directions

import torch
from collections import defaultdict
from tqdm import tqdm

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  PROCESS_HIDDEN_STATES_MODE = str(ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN)
  sys.argv = [
    'main.py',
    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', 'huginn-0125',

    '--huginn_num_steps', '32',

    '--data_path', f'{WORKSPACE_PATH}/datasets/lirefs',
    '--data_name', 'mmlu-pro-3000samples.json',

    '--data_sample_size', '24',
    '--data_batch_size', '1',

    '--process_hidden_states_mode', PROCESS_HIDDEN_STATES_MODE,

    '--output_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions',
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

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
  device_map=args['device_map'],
)

# %%

if args['data_name'].endswith('mmlu-pro-3000samples.json'):
  print(f"Loading dataset from JSON file {args['data_path']}/{args['data_name']} and sample size {args['data_sample_size']}")
  sampled_data = load_json_dataset(
    file_path=os.path.join(args['data_path'], args['data_name']),
    sample_size=args['data_sample_size'],
  )
else:
  raise ValueError(
    f"Unsupported data name: {args['data_name']}. "
    "Currently only JSON files are supported."
  )

# %%

print("Computing memory reasoning scores for the sampled data.")
reasoning_indices = [
  index for index, sample in enumerate(sampled_data) 
  if sample['memory_reason_score'] > 0.5
]
memorizing_indices = [
  index for index, sample in enumerate(sampled_data) 
  if sample['memory_reason_score'] <= 0.5
]

reasoning_indices_batched = [
  reasoning_indices[i:i + args['data_batch_size']]
  for i in range(
    0, len(reasoning_indices), args['data_batch_size']
  )
]

memorizing_indices_batched = [
  memorizing_indices[i:i + args['data_batch_size']]
  for i in range(
    0, len(memorizing_indices), args['data_batch_size']
  )
]

del reasoning_indices
del memorizing_indices

# %%

print("Preparing queries for the model.")
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

queries_reasoning_batched: list[list[str]] = [
  [] for _ in range(len(reasoning_indices_batched))
]
for batch_index, reasoning_indices_batch in enumerate(reasoning_indices_batched):
  for query_index in reasoning_indices_batch:
    queries_reasoning_batched[batch_index].append(queries[query_index])

queries_memorizing_batched = [
  [] for _ in range(len(memorizing_indices_batched))
]
for batch_index, memorizing_indices_batch in enumerate(memorizing_indices_batched):
  for query_index in memorizing_indices_batch:
    queries_memorizing_batched[batch_index].append(queries[query_index])

del queries

# %

del sampled_data

# %%

candidate_directions: defaultdict[str, CandidateDirectionStats] = None
print("Computing candidate directions for reasoning and memorizing indices.")
match args['process_hidden_states_mode']:
  case ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN:
    def process(
      queries_batched: list[list[str]], 
      label: str,
      candidate_directions: defaultdict[str, CandidateDirectionStats] | None,
    ):
      for queries_batch in tqdm(queries_batched):
        inputs = tokenize_text(
          model=model,
          tokenizer=tokenizer,
          text=queries_batch,
        )

        with torch.no_grad():
          hidden_states = get_hidden_states(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
          )

        # Processing hidden states
        processed_hidden_states = process_hidden_states(
          model=model,
          mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
          hidden_states=hidden_states,
        )

        # Computing candidate directions for reasoning and memorizing indices.
        candidate_directions = compute_candidate_directions(
          model=model,
          hidden_states=processed_hidden_states,
          label=label,
          candidate_directions=candidate_directions,
        )

        del inputs
        del hidden_states
        del processed_hidden_states
        torch.cuda.empty_cache()
      
      return candidate_directions
    
    candidate_directions = process(
      queries_batched=queries_reasoning_batched, 
      label="reasoning",
      candidate_directions=candidate_directions,
    )
    candidate_directions = process(
      queries_batched=queries_memorizing_batched, 
      label="memorizing",
      candidate_directions=candidate_directions,
    )
  case ProcessHiddenStatesMode.ALL_TOKENS:
    def process(
      queries_batched: list[list[str]],
      label: str,
      candidate_directions: defaultdict[str, CandidateDirectionStats] | None,
    ):
      for queries_batch in tqdm(queries_batched):
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
            huginn_num_steps=args.get('huginn_num_steps', 32),
          )

        responses = tokenizer.batch_decode(
          outputs,
          skip_special_tokens=True,
        )

        # Process hidden states for the responses
        inputs = tokenize_text(
          model=model,
          tokenizer=tokenizer,
          text=responses,
        )

        with torch.no_grad():
          hidden_states = get_hidden_states(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
          )

        # Processing hidden states
        processed_hidden_states = process_hidden_states(
          model=model,
          mode=ProcessHiddenStatesMode.ALL_TOKENS,
          hidden_states=hidden_states,
          attention_mask=inputs["attention_mask"],
        )

        # Computing candidate directions for reasoning and memorizing indices
        candidate_directions = compute_candidate_directions(
          model=model,
          hidden_states=processed_hidden_states,
          label=label,
          candidate_directions=candidate_directions,
        )

        del inputs
        del outputs
        del responses
        del hidden_states
        del processed_hidden_states
        torch.cuda.empty_cache()

      return candidate_directions
    
    candidate_directions = process(
      queries_batched=queries_reasoning_batched, 
      label="reasoning",
      candidate_directions=candidate_directions,
    )
    candidate_directions = process(
      queries_batched=queries_memorizing_batched,
      label="memorizing",
      candidate_directions=candidate_directions,
    )
  case _:
    raise ValueError(f"Unsupported cache hidden states mode: {args['process_hidden_states_mode']}")

# %%

print("Converting candidate directions from defaultdict to dict.")
candidate_directions = convert_defaultdict_to_dict(candidate_directions)

# %%

os.makedirs(args['output_path'], exist_ok=True)
candidate_directions_file_path = os.path.join(
  args['output_path'],
  f"{args['model_name']}_{args['data_name']}_{args['process_hidden_states_mode']}_candidate_directions.pt"
)
print(f"Saving candidate directions to {candidate_directions_file_path}")
torch.save(candidate_directions, candidate_directions_file_path)

# %%
