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
from utils import SaveHiddenStatesOutput
from utils import SaveHiddenStatesQueryLabel

import torch as t
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float
import matplotlib.pyplot as plt

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/media/npu-tao/disk4T/jason"
  sys.argv = [
    'main.py',
    '--device', 'cuda:0',

    '--models_name', 
    'Meta-Llama-3-8B',
    'huginn-0125',

    '--hidden_states_file_paths', 
    f'{WORKSPACE_PATH}/experiments/save_hidden_states/Meta-Llama-3-8B_hidden_states_FIRST_ANSWER_TOKEN.pt',
    f'{WORKSPACE_PATH}/experiments/save_hidden_states/huginn-0125_hidden_states_FIRST_ANSWER_TOKEN.pt',

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

print("Loading hidden states from file.")
hs_dict: dict[str, SaveHiddenStatesOutput] = {}
for model_name, hidden_states_file_path in zip(
  args['models_name'],
  args['hidden_states_file_paths'],
):
  print(f"Loading hidden states for model: {model_name} from {hidden_states_file_path}")
  
  # Load the hidden states output
  hs_dict[model_name] = t.load(
    hidden_states_file_path,
    map_location="cpu",
    weights_only=False,
  )

# %%

print("Collecting reasoning and memorizing indices.")
reasoning_indices_dict: dict[str, list[int]] = {}
memorizing_indices_dict: dict[str, list[int]] = {}
for model_name, hs_output in hs_dict.items():
  print(f"Processing model: {model_name}")
  
  # Collect reasoning and memorizing indices
  reasoning_indices_dict[model_name] = [
    i for i, label in enumerate(hs_output['labels']) 
    if label == SaveHiddenStatesQueryLabel.REASONING.value
  ]
  memorizing_indices_dict[model_name] = [
    i for i, label in enumerate(hs_output['labels'])
    if label == SaveHiddenStatesQueryLabel.MEMORIZING.value
  ]

# %%

print("Computing candidate directions.")
candidate_directions_dict: dict[str, dict[str, list[Float[Tensor, "n_embd"]]]] = {}
for model_name, hs_output in hs_dict.items():
  print(f"Computing candidate directions for model: {model_name}")
  candidate_directions_dict[model_name] = {
    "reasoning": [],
    "memorizing": [],
    "results": {}
  }
  for layer_index, hs in hs_output['hidden_states'].items():
    hs_tensor = t.stack(hs, dim=0)

    reasoning_indices = reasoning_indices_dict[model_name]
    memorizing_indices = memorizing_indices_dict[model_name]
    
    hs_reasoning = hs_tensor[reasoning_indices].to(dtype=t.float64)
    hs_memorizing = hs_tensor[memorizing_indices].to(dtype=t.float64)

    candidate_directions_dict[model_name]["reasoning"].append(hs_reasoning.mean(dim=0))
    candidate_directions_dict[model_name]["memorizing"].append(hs_memorizing.mean(dim=0))

# %%

directions: dict[
  str, 
  list[Float[Tensor, "n_embd"]]
] = {}
for model_name, candidate_directions in candidate_directions_dict.items():
  print(f"Computing directions for model: {model_name}")
  directions[model_name] = []
  for layer_index in range(len(candidate_directions["reasoning"])):
    reasoning_direction = candidate_directions["reasoning"][layer_index]
    memorizing_direction = candidate_directions["memorizing"][layer_index]
    
    directions[model_name].append(
      reasoning_direction - memorizing_direction
    )

# %%

print("Computing cosine similarities between hidden states and candidate directions.")
cos_sim_dict: dict[
  str, 
  dict[str, list[Float[Tensor, "batch_size"]]]
] = {}
for model_name, hs_output in hs_dict.items():
  cos_sim_dict[model_name] = {
    "reasoning set": [],
    "memorizing set": [],
  }
  for layer_index, hs in hs_output['hidden_states'].items():
    """
    ```python
    hs_reasoning = t.tensor(
      [
        [0,1],
        [1,0]
      ], 
      dtype=t.float32
    )
    candidate_direction = t.tensor([0,1], dtype=t.float32)

    F.cosine_similarity(x1=hs_reasoning, x2=candidate_direction.unsqueeze(0), dim=-1)
    ```
    """
    hs_tensor = t.stack(hs, dim=0)

    hs_reasoning = hs_tensor[reasoning_indices]
    hs_memorizing = hs_tensor[memorizing_indices]

    cos_sim_dict[model_name]["reasoning set"].append(F.cosine_similarity(
      hs_reasoning,
      directions[model_name][layer_index].unsqueeze(0),
      dim=-1,
    ))
    cos_sim_dict[model_name]["memorizing set"].append(F.cosine_similarity(
      hs_memorizing,
      directions[model_name][layer_index].unsqueeze(0),
      dim=-1,
    ))

  def get_layer_index(
    cos_sim: list[Float[Tensor, "batch_size"]]
  ) -> int:
    mean = [cos_sim_i.mean() for cos_sim_i in cos_sim]
    max_index = mean.index(max(mean))
    min_index = mean.index(min(mean))
    return max_index, min_index

  res_res_max, res_res_min = get_layer_index(cos_sim_dict[model_name]['reasoning set'])
  res_mem_max, res_mem_min = get_layer_index(cos_sim_dict[model_name]['memorizing set'])

  print(f"Model {model_name}")
  print(f"Reasoning direction @ Reasoning set: max at layer {res_res_max} max, min at layer {res_res_min}")
  print(f"Reasoning direction @ Memorizing set: max at layer {res_mem_max}, min at layer {res_mem_min}")

  candidate_directions_dict[model_name]['results'].setdefault('cosine similarity', {})
  candidate_directions_dict[model_name]['results']['cosine similarity'].update({
    'reasoning direction @ reasoning set max layer': res_res_max,
    'reasoning direction @ reasoning set min layer': res_res_min,
  })
  candidate_directions_dict[model_name]['results']['cosine similarity'].update({
    'reasoning direction @ memorizing set max layer': res_mem_min,
    'reasoning direction @ memorizing set min layer': res_mem_max,
  })

# %%

os.makedirs(args['output_path'], exist_ok=True)
candidate_directions_file_path = os.path.join(
  args['output_path'],
  f"candidate_directions_FIRST_ANSWER_TOKEN.pt"
)
print(f"Saving candidate directions to {candidate_directions_file_path}")
t.save(
  candidate_directions_dict, 
  candidate_directions_file_path
)

# %%

nrows = 1
ncols = len(cos_sim_dict)

fig, axes = plt.subplots(
  nrows=nrows,
  ncols=ncols,
  figsize=(6.48 * ncols * 1.5, 4.8 * nrows * 1.5),
)
axes = axes.flatten()

for ax, (model_name, cos_sim) in zip(axes, cos_sim_dict.items()):

  def plot(
    ax, 
    cos_sim: list[Float[Tensor, "batch_size"]],
    label: str
  ):
    cos_sim_tensor = t.stack(cos_sim)
    mean = cos_sim_tensor.mean(dim=-1)
    std = cos_sim_tensor.std(dim=-1)

    layers = list(range(cos_sim_tensor.shape[0]))

    ax.plot(
      layers, 
      mean, 
      label=label,
    )
    ax.fill_between(
      x=layers, 
      y1=mean - std, 
      y2=mean + std,
      alpha=0.2,
    )
  
  plot(
    ax,
    cos_sim['reasoning set'],
    label="Reasoning direction @ Reasoning Set",
  )
  plot(
    ax,
    cos_sim['memorizing set'],
    label="Reasoning direction @ Memorizing Set",
  )

  ax.set_xlabel("Layer Index")
  ax.set_ylabel("Cosine Similarity")
  ax.set_title(model_name)
  ax.grid(True)
  ax.legend()

# %%

cosine_similarities_plot_path = os.path.join(
  args['output_path'],
  f"cosine_similarities_FIRST_ANSWER_TOKEN.png"
)
fig.savefig(cosine_similarities_plot_path, bbox_inches='tight')
print(f"Cosine similarities plot saved to {cosine_similarities_plot_path}")

# %%
