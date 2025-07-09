from utils import get_device_map
from utils import CandidateDirectionStats

import torch
from torch import Tensor
from jaxtyping import Float
from enum import Enum

class DirectionNormalizationMode(Enum):
  """
  Enum for different methods of computing candidate directions.

  - UNIT_VECTOR: Computes the unit vector of the difference in mean between 
    the positive set (reasoning) and negative set (memorizing).

  - SCALE_WITH_OVERALL_MAGNITUDE: Scale the candidate direction with
    the magnitude of the combined positive (reasoning) set and negative set (memorizing).
    The purpose is to ensure that all candidate directions have a consistent magnitude,
    matching that of the mean activation vector of the combined positive and negative sets.
    This prevents features with naturally larger or smaller difference vectors from having a 
    disproportionate effect when they're used for steering the model's behavior later on. 
    It standardizes their "strength" while keeping their unique directional information.

  Notes:
  - Magnitude (or norm) is the length of a vector.
  - A unit vector is a vector whose mangitude (length) is exactly 1.
    Its purpose is to represent a direction

  Reference:
  [1] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/Intervention/utils.py#L247
  [2] https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/utils/utils.py#L267-L270 
  """
  UNIT_VECTOR = "UNIT_VECTOR"
  SCALE_WITH_OVERALL_MAGNITUDE = "SCALE_WITH_OVERALL_MAGNITUDE"

  def __str__(self):
    return self.value

def compute_directions(
  model,
  candidate_directions: dict[str, CandidateDirectionStats],
  positive_label: str,
  negative_label: str,
  overall_label: str,
  normalization_mode: DirectionNormalizationMode = DirectionNormalizationMode.UNIT_VECTOR,
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
    overall_candidate_direction = candidate_directions[overall_label]["mean"][layer_index].to(device=device, dtype=torch.float64)
    direction = positive_candidate_direction - negative_candidate_direction

    match normalization_mode:
      case DirectionNormalizationMode.UNIT_VECTOR:
        direction_magnitude = direction.norm(dim=-1, keepdim=True) + 1e-8 # Adding a small value to avoid division by zero
        direction = direction / direction_magnitude # Normalize the direction
      case DirectionNormalizationMode.SCALE_WITH_OVERALL_MAGNITUDE:
        direction_magnitude = direction.norm(dim=-1, keepdim=True) + 1e-8
        direction_unit_vector = direction / direction_magnitude
        overall_magnitude = overall_candidate_direction.norm(dim=-1, keepdim=True) + 1e-8
        direction = direction_unit_vector * overall_magnitude
      case _:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode}")
    
    directions[layer_index] = direction.to(device=device, dtype=model.dtype)
    
    del positive_candidate_direction
    del negative_candidate_direction
    del overall_candidate_direction
    del direction
  torch.cuda.empty_cache()
  
  return directions