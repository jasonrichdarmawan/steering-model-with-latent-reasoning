from utils import get_device_map
from utils import get_n_embd

from typing import TypedDict
from jaxtyping import Float
import torch
from torch import Tensor
from collections import defaultdict

class CandidateDirectionStats(TypedDict):
  mean: defaultdict[int, Float[Tensor, "n_embd"]]
  count: defaultdict[int, int]

def compute_candidate_directions(
  model,
  hidden_states: dict[int, list[Float[Tensor, "seq_len n_embd"]]],
  label: str,
  candidate_directions: defaultdict[str, CandidateDirectionStats] | None = None,
):
  """
  Mean_n     =   "Sum"_{n}                                                / n
  Mean_{n+1} = ( "Sum"_{n}  + "new_item" )                                / (n              +1)
  
  Equivalent to

  "new_mean" = "current_mean"           + ("new_item" - "current_mean")  / ("current_count" +1)
  Mean_{n+1} = "Sum"_{n} / n            + ("new_item" - "Sum"_{n} / n)   / (n               +1)
             = ( "Sum"_{n} / n  * (n+1) + ("new_item" - "Sum"_{n} / n) ) / (n               +1)
             = ( "Sum"_{n} + "new_item" )                                / (n               +1)
  """
  if candidate_directions is None:
    n_embd = get_n_embd(model=model)
    candidate_directions = defaultdict(
      lambda: {
        "mean": defaultdict(
          lambda: torch.zeros(
            n_embd, 
            dtype=torch.float64, 
            device="cuda",
          )
        ),
        "count": defaultdict(int),
      }
    )

  match model.config.model_type:
    case "huginn_raven" | "llama":
      # Build a mapping from layer index to CUDA device id
      device_map = get_device_map(model=model)
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
  
  n_batches_to_cache = range(len(hidden_states[0]))
  for layer_index, device in device_map.items():
    hidden_states_layer = hidden_states[layer_index]
    for batch_index in n_batches_to_cache:
      new_item = hidden_states_layer[batch_index].mean(dim=0)

      # Update label-specific mean and count
      label_current_mean = candidate_directions[label]["mean"][layer_index].to(device=device)
      label_current_count = candidate_directions[label]["count"][layer_index]
      label_new_mean = (
        label_current_mean 
        + (new_item - label_current_mean) 
          / (label_current_count + 1)
      )
      candidate_directions[label]["mean"][layer_index] = label_new_mean
      candidate_directions[label]["count"][layer_index] += 1
  
      # Update overall mean and count
      overall_current_mean = candidate_directions["overall"]["mean"][layer_index].to(device=device)
      overall_current_count = candidate_directions["overall"]["count"][layer_index]
      overall_new_mean = (
        overall_current_mean
        + (new_item - overall_current_mean) 
          / (overall_current_count + 1)
      )
      candidate_directions["overall"]["mean"][layer_index] = overall_new_mean
      candidate_directions["overall"]["count"][layer_index] += 1

      del new_item
      del label_current_mean
      del overall_current_mean
  torch.cuda.empty_cache()
  
  return candidate_directions