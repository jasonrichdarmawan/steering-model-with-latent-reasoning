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
    n_embd = get_n_embd(model)
    candidate_directions = defaultdict(
      lambda: {
        "mean": defaultdict(
          lambda: torch.zeros(
            n_embd, 
            dtype=torch.float64, 
            device="cpu"
          )
        ),
        "count": defaultdict(int),
      }
    )
  
  n_batches_to_cache = range(len(hidden_states[0]))
  for layer_index, hidden_states_layer in hidden_states.items():
    for batch_index in n_batches_to_cache:
      new_item = hidden_states_layer[batch_index].to(device="cuda").mean(dim=0)

      # Update label-specific mean and count
      label_current_mean = candidate_directions[label]["mean"][layer_index].to(device="cuda")
      label_current_count = candidate_directions[label]["count"][layer_index]
      label_new_mean = (
        label_current_mean 
        + (new_item - label_current_mean) 
          / (label_current_count + 1)
      )
      candidate_directions[label]["mean"][layer_index] = label_new_mean.to(device="cpu")
      candidate_directions[label]["count"][layer_index] += 1
  
      # Update overall mean and count
      overall_current_mean = candidate_directions["overall"]["mean"][layer_index].to(device="cuda")
      overall_current_count = candidate_directions["overall"]["count"][layer_index]
      overall_new_mean = (
        overall_current_mean
        + (new_item - overall_current_mean) 
          / (overall_current_count + 1)
      )
      candidate_directions["overall"]["mean"][layer_index] = overall_new_mean.to(device="cpu")
      candidate_directions["overall"]["count"][layer_index] += 1

      del new_item
      del label_current_mean
      del overall_current_mean
      torch.cuda.empty_cache()
  
  return candidate_directions