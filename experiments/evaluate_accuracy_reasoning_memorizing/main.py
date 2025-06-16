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
  
  reload(sys.modules.get('earm_utils.parse_args', sys))
  reload(sys.modules.get('earm_utils', sys))

  reload(sys.modules.get('utils.use_deterministic_algorithms', sys))
  reload(sys.modules.get('utils.load_model_and_tokenizer', sys))
  reload(sys.modules.get('utils.load_hidden_states_cache', sys))
  reload(sys.modules.get('utils.load_json_dataset', sys))
  reload(sys.modules.get('utils.compute_candidate_directions', sys))
  reload(sys.modules.get('utils.prepare_fewshot_prompts', sys))
  reload(sys.modules.get('utils.prepare_queries', sys))
  reload(sys.modules.get('utils.set_activations_hooks', sys))
  reload(sys.modules.get('utils.remove_hooks', sys))
  reload(sys.modules.get('utils.generate_sentences', sys))
  reload(sys.modules.get('utils', sys))

from earm_utils import parse_args

from utils import use_deterministic_algorithms
from utils import load_model_and_tokenizer
from utils import load_hidden_states_cache
from utils import load_json_dataset
from utils import compute_candidate_directions
from utils import prepare_fewshot_prompts
from utils import prepare_queries
from utils import generate_sentences
from utils import ProjectionHookConfig
from utils import set_activations_hooks, remove_hooks
from utils import set_model_predict_correctness

from transformers import GenerationConfig
from tqdm import tqdm
import random
import torch

# %%

if False:
  import sys

  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/home/npu-tao/jason"
  sys.argv = [
    'main.py',
    '--models_path', f'{root_path}/transformers',
    '--model_name', 'huginn-0125',

    '--hidden_states_cache_path', f'{root_path}/experiments/hidden_states_cache',
    '--mmlu_pro_3000samples_data_file_path', f'{root_path}/datasets/mmlu-pro-3000samples.json',

    '--test_data_path', f'{root_path}/datasets',
    '--test_data_name', 'mmlu-pro-3000samples',
    '--with_fewshot_prompts',
    # '--with_cot',
    '--batch_size', '1',
    '--max_new_tokens', '200',

    '--with_intervention',
    '--layer_indices', '66',
    # '--with_pre_hook',
    '--with_post_hook',
    '--scale', '0.1'
  ]

args = parse_args()

# %%

print("Setting deterministic algorithms for reproducibility.")
use_deterministic_algorithms()

# %%

model, tokenizer = load_model_and_tokenizer(
  model_path=args['models_path'],
  model_name=args['model_name']
)

# %%

mmlu_pro_3000samples_dataset = load_json_dataset(
  file_path=args['mmlu_pro_3000samples_data_file_path'],
)

reasoning_indices = [
  index for index, sample in enumerate(mmlu_pro_3000samples_dataset) 
  if sample['memory_reason_score'] > 0.5
]
memorizing_indices = [
  index for index, sample in enumerate(mmlu_pro_3000samples_dataset) 
  if sample['memory_reason_score'] <= 0.5
]

# %%

if args['with_intervention']:
  print("Loading hidden states cache for intervention.")
  hidden_states_cache = load_hidden_states_cache(
    hidden_states_cache_path=args['hidden_states_cache_path'],
    model_name=args['model_name']
  )

  candidate_directions = compute_candidate_directions(
    hidden_states_cache=hidden_states_cache["mmlu-pro-3000samples"],
    reasoning_indices=reasoning_indices,
    memorizing_indices=memorizing_indices,
    dtype=model.dtype
  )
else:
  candidate_directions = None

# %%

# Load the test dataset based on the specified test data name.
# Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/main/reasoning_representation/Intervention/features_intervention.py
match args['test_data_name']:
  case 'mmlu-pro-3000samples':
    print("Loading MMLU-Pro 3000 samples dataset for testing.")
    test_dataset = random.sample(mmlu_pro_3000samples_dataset, 200)

    reasoning_indices = [index for index, sample in enumerate(test_dataset) if sample['memory_reason_score'] > 0.5]
    memorizing_indices = [index for index, sample in enumerate(test_dataset) if sample['memory_reason_score'] <= 0.5]

  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%

if args['with_fewshot_prompts']:
  print("Preparing few-shot prompts for the test dataset.")
  fewshot_prompts = prepare_fewshot_prompts(
    data_path=args['test_data_path'],
    data_name=args['test_data_name'],
    with_cot=args['with_cot'],
  )
else:
  fewshot_prompts = None

# %%

queries = prepare_queries(
  model_name=model.config.model_type,
  data=test_dataset,
  data_name=args['test_data_name'],
  tokenizer=tokenizer,
  system_prompt="You are a helpful assistant.",
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

def _compute_accuracy(
  entries: list[str],
  label: str
):
  total = len(entries)
  if total == 0:
    return None
  correct = sum(entry.get("model_predict_correctness", False) for entry in entries)
  accuracy = correct / total
  print(f"{label} Accuracy: {accuracy:.4f} ({correct}/{total})")
  return accuracy

if args['with_intervention']:
  projection_hook_config = ProjectionHookConfig(
    layer_indices=args['layer_indices'],
    candidate_directions=candidate_directions,
    pre_hook=args['with_pre_hook'],
    post_hook=args['with_post_hook'],
    scale=args['scale']
  )

  hooks = set_activations_hooks(
    model=model,
    candidate_directions=candidate_directions,
    config=projection_hook_config,
  )

match model.config.model_type:
  case name if name.startswith("huginn_"):
    """
    Reference:
    [1] https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
    [2] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/main/reasoning_representation/Intervention/features_intervention.py
    """
    generation_config = GenerationConfig(
      max_new_tokens=args['max_new_tokens'],
      stop_strings=["<|end_text|>", "<|end_turn|>"],
      do_sample=False,
      temperature=None,
      top_p=None,
      min_p=None,
      return_dict_in_generate=True,
      eos_token_id=65505,
      bos_token_id=65504,
      pad_token_id=65509
    )

    for queries_batch, entries_batch in tqdm(
      zip(queries_batched, entries_batched),
      total=len(queries_batched)
    ):
      responses = generate_sentences(
        model=model,
        tokenizer=tokenizer,
        text=queries_batch,
        config=generation_config,
      )

      set_model_predict_correctness(
        entries=entries_batch,
        queries=queries_batch,
        responses=responses,
        test_dataset_name=args['test_data_name']
      )

      torch.cuda.empty_cache()
  case _:
    raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
if args['with_intervention']:
  remove_hooks(hooks)

match args['test_data_name']:
  case 'mmlu-pro-3000samples':
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

    _compute_accuracy(
      entries=reasoning_entries,
      label=f"Layer {args['layer_indices']} - Reasoning Subset"
    )
    _compute_accuracy(
      entries=memorizing_entries,
      label=f"Layer {args['layer_indices']} - Memorizing Subset"
    )
  case _:
    raise ValueError(f"Unsupported test data name: {args['test_data_name']}")

# %%
