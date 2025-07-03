from utils import get_n_layers

import torch
from torch import Tensor
from jaxtyping import Float

def compute_candidate_directions(
  model,
  hidden_states_cache: dict[int, Float[Tensor, "batch n_embd"]],
  reasoning_indices: list[int],
  memorizing_indices: list[int],
  layer_indices: list[int],
) -> dict[int, Float[Tensor, "n_layers n_embd"]]:
  """
  Tensor.to(torch.float64) is used because we 
  follow the [original implementation](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py).
  The reason is to avoid numerical instability 
  when computing the mean of the hidden states.
  """
  candidate_directions: dict[int, Float[Tensor, "n_layers n_embd"]] = {}

  match model.config.model_type:
    case name if name.startswith("huginn_") or "llama":
      # Build a mapping from layer index to CUDA device id
      device_map = get_device_map(model=model)
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
  
  for layer_index, device in {
    k: v for k, v in device_map.items()
    if k in layer_indices
  }.items():
    reasoning_hidden_states = hidden_states_cache[layer_index][reasoning_indices].to(device=device, dtype=torch.float64)
    memorizing_hidden_states = hidden_states_cache[layer_index][memorizing_indices].to(device=device, dtype=torch.float64)

    reasoning_mean = reasoning_hidden_states.mean(dim=0)
    memorization_mean = memorizing_hidden_states.mean(dim=0)

    candidate_direction = reasoning_mean - memorization_mean
    magnitude = candidate_direction.norm(dim=-1, keepdim=True) + 1e-8  # Adding a small value to avoid division by zero
    candidate_direction = candidate_direction / magnitude  # Normalize the direction
    candidate_direction = candidate_direction.to(device=device, dtype=model.dtype)

    candidate_directions[layer_index] = candidate_direction

  if len(candidate_directions) == 0:
    raise ValueError("No candidate directions were computed. Check the provided layer indices and hidden states cache.")
  
  return candidate_directions

def get_device_map(
  model,
) -> dict[int, int]:
  device_map: dict[int, int] = {}

  match model.config.model_type:
    case name if name.startswith("huginn_"):
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