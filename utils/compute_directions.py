from utils import get_device_map
from utils import CandidateDirectionStats

import torch
from torch import Tensor
from jaxtyping import Float

def compute_directions(
  model,
  candidate_directions: dict[str, CandidateDirectionStats],
  positive_label: str,
  negative_label: str,
) -> dict[int, Float[Tensor, "n_embd"]]:
  """
  Assumming the hidden_states.dtype is torch.float32 or smaller,
  torch.float64 is used to avoid numerical instability 
  when computing the mean of the hidden states.

  Reference:
  1. https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  directions: dict[int, Float[Tensor, "n_embd"]] = {}

  match model.config.model_type:
    case "huginn_raven" | "llama":
      # Build a mapping from layer index to CUDA device id
      device_map = get_device_map(model=model)
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")

  for layer_index, device in device_map.items():
    positive_candidate_direction = candidate_directions[positive_label]["mean"][layer_index].to(device=device, dtype=torch.float64)
    negative_candidate_direction = candidate_directions[negative_label]["mean"][layer_index].to(device=device, dtype=torch.float64)
    direction = positive_candidate_direction - negative_candidate_direction
    
    directions[layer_index] = direction.to(device=device, dtype=model.dtype)
    
    del positive_candidate_direction
    del negative_candidate_direction
    # del overall_candidate_direction
    del direction
  torch.cuda.empty_cache()
  
  return directions