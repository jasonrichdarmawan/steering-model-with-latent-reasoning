import torch
from torch import Tensor
from jaxtyping import Float

def compute_candidate_directions(
  hidden_states_cache: dict[int, Float[Tensor, "batch n_embd"]],
  reasoning_indices: list[int],
  memorizing_indices: list[int],
  dtype: torch.dtype
):
  """
  Tensor.to(torch.float64) is used because we 
  follow the [original implementation](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py).
  The reason is to avoid numerical instability 
  when computing the mean of the hidden states.
  """
  n_layers = len(hidden_states_cache)
  n_embd = hidden_states_cache[0].shape[-1]

  candidate_directions = torch.zeros(
    (n_layers, n_embd), dtype=torch.float64, 
    device="cuda"
  )
  
  for layer_index in range(n_layers):
    reasoning_hidden_states = hidden_states_cache[layer_index][reasoning_indices].to(torch.float64)
    memorizing_hidden_states = hidden_states_cache[layer_index][memorizing_indices].to(torch.float64)

    reasoning_mean = reasoning_hidden_states.mean(dim=0)
    memorization_mean = memorizing_hidden_states.mean(dim=0)

    candidate_direction = reasoning_mean - memorization_mean
    magnitude = candidate_direction.norm(dim=-1, keepdim=True) + 1e-8  # Adding a small value to avoid division by zero
    candidate_direction = candidate_direction / magnitude  # Normalize the direction

    candidate_directions[layer_index] = candidate_direction
  
  return candidate_directions.to(dtype=dtype) # Convert to the model's dtype