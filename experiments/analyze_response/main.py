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

from ar_utils import parse_args

from utils import enable_reproducibility
from utils import load_model_and_tokenizer
from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode
from utils import prepare_query
from utils import get_device_map
from utils import ProjectionHookConfig
from utils import ProjectionHookConfigLiReFs
from utils import set_activations_hooks
from utils import tokenize_text
from utils import generate

from jaxtyping import Float
import torch as t
from torch import Tensor
from enum import Enum
from typing import TypedDict
from tqdm import tqdm

# %%

if False:
  import sys

  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  MODEL_NAME = "Meta-Llama-3-8B"

  DIRECTION_NORMALIZATION_MODE = DirectionNormalizationMode.UNIT_VECTOR
  PROJECTION_HOOK_MODE = ProjectionHookMode.FEATURE_ADDITION
  MODIFICATION_MODE = TokenModificationMode.LAST_TOKEN

  sys.argv = [
    'main.py',

    '--models_path', f'{WORKSPACE_PATH}/transformers',
    '--model_name', MODEL_NAME,

    '--batch_size', '1',

    '--huginn_num_steps', '32',

    '--use_linear_probes',
    '--linear_probes_file_path', f'{WORKSPACE_PATH}/experiments/train_linear_probes/{MODEL_NAME}/best_checkpoint.pt',

    # '--use_candidate_directions',
    # '--candidate_directions_file_path', f'{WORKSPACE_PATH}/experiments/save_candidate_directions/candidate_directions_FIRST_ANSWER_TOKEN.pt',

    '--layer_indices', '12',
    '--direction_normalization_mode', str(DIRECTION_NORMALIZATION_MODE),
    '--projection_hook_mode', str(PROJECTION_HOOK_MODE),
    '--modification_mode', str(MODIFICATION_MODE),

    # '--with_hidden_states_pre_hook',
    '--with_hidden_states_post_hook',

    # '--with_attention_pre_hook',
    # '--with_attention_post_hook',

    # '--with_mlp_pre_hook',
    # '--with_mlp_post_hook',

    '--scale', '1',
  ]

args = parse_args()
print(f"Parsed arguments:")
print("#" * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Setting deterministic algorithms for reproducibility.")
enable_reproducibility()

# %%

model, tokenizer = load_model_and_tokenizer(
  models_path=args['models_path'],
  model_name=args['model_name'],
)

# %%

print("Preparing queries")

questions = [
  "What causes the Earth's seasons?",
]

queries = [
  prepare_query(
    model_name=model.config.model_type,
    question_content=question,
    tokenizer=tokenizer,
    apply_question_format=False,
    apply_chat_template=False,
    system_prompt=None,
    fewshot_prompts=None,
    with_cot=False,
  )
  for question in questions
]

queries_sorted = sorted(
  queries,
  key=lambda x: len(x),
  reverse=True,
)

queries_batched = [
  queries_sorted[i:i + args['batch_size']]
  for i in range(0, len(queries_sorted), args['batch_size'])
]

del queries
del queries_sorted

# %%

directions: dict[int, Float[Tensor, "n_embd"]] = {}
if args['use_linear_probes']:
  print("Using linear probes for the intervention.")
  checkpoint = t.load(
    args['linear_probes_file_path'],
    map_location='cpu',
    weights_only=False,
  )
  print("Best layer index:", checkpoint.get('layer_index', None))

  directions = {
    layer_index: linear_probe[:, 0] - linear_probe[:, 1]
    for layer_index, linear_probe in enumerate(checkpoint['linear_probe'])
  }

  del checkpoint
elif args['use_candidate_directions']:
  print("Computing directions for the model.")
  candidate_directions = t.load(
    args['candidate_directions_file_path'],
    map_location='cpu',
    weights_only=False,
  )[args['model_name']]

  for layer_index in range(len(candidate_directions['reasoning'])):
    directions[layer_index] = candidate_directions['reasoning'][layer_index] - candidate_directions['memorizing'][layer_index]

  del candidate_directions
else:
  raise ValueError(
    "Either 'use_linear_probes' or 'use_candidate_directions' must be True when 'with_intervention' is True."
  )

device_map = get_device_map(model=model)

for layer_index, direction in directions.items():
  directions[layer_index] = direction.to(
    device=device_map[layer_index],
    dtype=model.dtype,
  )

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
  case "llama":
    projection_hook_config = ProjectionHookConfigLiReFs(
      steering_mode=args['projection_hook_mode'],
      modification_mode=args['modification_mode'],
      direction_normalization_mode=args['direction_normalization_mode'],
      layer_indices=args['layer_indices'],
      hidden_states_hooks_config={
        "pre_hook": args['with_hidden_states_pre_hook'],
        "post_hook": args['with_hidden_states_post_hook'],
      },
      attention_hooks_config={
        "pre_hook": args['with_attention_pre_hook'],
        "post_hook": args['with_attention_post_hook'],
      },
      mlp_hooks_config={
        "pre_hook": args['with_mlp_pre_hook'],
        "post_hook": args['with_mlp_post_hook'],
      },
      scale=args['scale'],
    )
  case _:
    raise ValueError(f"Unsupported model type: {model.config.model_type}")

class Mode(Enum):
  WITHOUT_INTERVENTION = "WITHOUT_INTERVENTION"
  WITH_INTERVENTION = "WITH_INTERVENTION"

class Response(TypedDict):
  without_intervention: str
  with_intervention: str

responses_batched: list[list[Response]] = []
for queries_batch in tqdm(queries_batched):

  responses_batch_mode: dict[Mode, list[str]] = {}

  for mode in [Mode.WITHOUT_INTERVENTION, Mode.WITH_INTERVENTION]:
    if mode == Mode.WITH_INTERVENTION:
      hooks = set_activations_hooks(
        model=model,
        feature_directions=directions,
        config=projection_hook_config,
        verbose=False,
      )

    inputs = tokenize_text(
      model=model,
      tokenizer=tokenizer,
      text=queries_batch,
    )

    with t.no_grad():
      outputs = generate(
        model=model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        tokenizer=tokenizer,
        huginn_num_steps=args.get("huginn_num_steps", None),
      )

    responses_batch = tokenizer.batch_decode(
      outputs[:, inputs["input_ids"].shape[1]:],
      skip_special_tokens=True,
    )

    responses_batch_mode[mode] = responses_batch

    if mode == Mode.WITH_INTERVENTION:
      for hook in hooks:
        hook.remove()
      hooks.clear()

    del inputs
  
  responses_batch_formatted: list[Response] = [
    Response(
      without_intervention=responses_batch_mode[Mode.WITHOUT_INTERVENTION][i],
      with_intervention=responses_batch_mode[Mode.WITH_INTERVENTION][i]
    )
    for i in range(len(queries_batch))
  ]
  responses_batched.append(responses_batch_formatted)

for queries_batch, responses_batch in zip(queries_batched, responses_batched):
  for query, response in zip(queries_batch, responses_batch):
    print("#" * 60)

    print("Query:")
    print(query)

    print("#" * 30)

    print("Response without intervention:")
    print(response["without_intervention"])

    print("#" * 30)

    print("Response with intervention:")
    print(response["with_intervention"])
  
    print("#" * 60)

# %%
