from utils import get_n_layers
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
  Tensor.to(torch.float64) is used because we 
  follow the [original implementation](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py).
  The reason is to avoid numerical instability 
  when computing the mean of the hidden states.
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

    # reasoning_hidden_states = hidden_states_cache[layer_index][reasoning_indices].to(device=device, dtype=torch.float64)
    # memorizing_hidden_states = hidden_states_cache[layer_index][memorizing_indices].to(device=device, dtype=torch.float64)

    # reasoning_mean = reasoning_hidden_states.mean(dim=0)
    # memorization_mean = memorizing_hidden_states.mean(dim=0)

    # candidate_direction = reasoning_mean - memorization_mean

    # match normalization_mode:
    #   case NormalizationMode.UNIT_VECTOR:
    #     pass
    #     difference_in_mean_magnitude = candidate_direction.norm(dim=-1, keepdim=True) + 1e-8 # Adding a small value to avoid division by zero
    #     candidate_direction = candidate_direction / difference_in_mean_magnitude # Normalize the direction
    #   case NormalizationMode.SCALE_WITH_POSITIVE_NEGATIVE_SET_MAGNITUDE:
    #     difference_in_mean_magnitude = candidate_direction.norm(dim=-1, keepdim=True) + 1e-8
    #     candidate_direction_unit_vector = candidate_direction / difference_in_mean_magnitude
    #     overall_magnitude = torch.cat(
    #       (reasoning_hidden_states, memorizing_hidden_states), dim=0
    #     ).mean(dim=0).norm(dim=-1, keepdim=True) + 1e-8
    #     candidate_direction = candidate_direction_unit_vector * overall_magnitude
    #   case _:
    #     raise ValueError(f"Unsupported normalization mode: {normalization_mode}")
    
    # candidate_direction = candidate_direction.to(device=device, dtype=model.dtype)

    # candidate_directions[layer_index] = candidate_direction

  # if len(candidate_directions) == 0:
  #   raise ValueError("No candidate directions were computed. Check the provided layer indices and hidden states cache.")
  
  return directions

def get_device_map(
  model,
) -> dict[int, int]:
  device_map: dict[int, int] = {}

  match model.config.model_type:
    case "huginn_raven":
      n_layers = get_n_layers(model)

      if len(model.hf_device_map) == 1:
        # If the model is on a single device, assign all layers to that device
        device = model.hf_device_map['']
        for i in range(n_layers):
          device_map[i] = device
        return device_map

      recurrent_block_end_index = (
        model.config.n_layers_in_prelude
        + model.config.n_layers_in_recurrent_block
        * model.config.mean_recurrence
      )
      ln_f_index = (
        model.config.effective_expected_depth
      )

      for i in range(n_layers):
        # Prelude layers
        if i < model.config.n_layers_in_prelude:
          device_map[i] = model.hf_device_map["transformer.prelude"]

        # Core block layers (recurrent, may repeat device assignment)
        elif i < recurrent_block_end_index:
          device_map[i] = model.hf_device_map[
            f"transformer.core_block.{(i - model.config.n_layers_in_prelude) % model.config.n_layers_in_recurrent_block}"
          ]

        # Coda layers
        elif i < model.config.effective_expected_depth:
          device_map[i] = model.hf_device_map["transformer.coda"]

        elif i == ln_f_index:
          device_map[i] = model.hf_device_map["transformer.ln_f"]
        
        else:
          raise ValueError(f"Layer index {i} exceeds the expected number of layers {n_layers} for Huginn model.")

      return device_map
    case "llama":
      n_layers = get_n_layers(model)
      if len(model.hf_device_map) == 1:
        # If the model is on a single device, assign all layers to that device
        device = model.hf_device_map['']
        for i in range(n_layers):
          device_map[i] = device
        return device_map
      
      for i in range(n_layers):
        if i < model.config.num_hidden_layers:
          # Assign each layer to the device specified in the hf_device_map
          device_map[i] = model.hf_device_map[f"model.layers.{i}"]
        elif i == model.config.num_hidden_layers:
          device_map[i] = model.hf_device_map["model.norm"]
        else:
          raise ValueError(f"Layer index {i} exceeds the expected number of layers {n_layers} for Llama model.")
      return device_map
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")