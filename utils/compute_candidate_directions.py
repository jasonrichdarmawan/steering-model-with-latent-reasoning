import torch
from torch import Tensor
from jaxtyping import Float

def compute_candidate_directions(
  model,
  hidden_states_cache: dict[int, Float[Tensor, "batch n_embd"]],
  reasoning_indices: list[int],
  memorizing_indices: list[int],
) -> dict[int, Float[Tensor, "n_layers n_embd"]]:
  """
  Tensor.to(torch.float64) is used because we 
  follow the [original implementation](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py).
  The reason is to avoid numerical instability 
  when computing the mean of the hidden states.
  """
  candidate_directions = {}

  match model.config.model_type:
    case name if name.startswith("huginn_"):
      # Build a mapping from layer index to CUDA device id
      device_map = get_device_map(model)
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
  
  for layer_index, device in device_map.items():
    reasoning_hidden_states = hidden_states_cache[layer_index][reasoning_indices].to(device=device, dtype=torch.float64)
    memorizing_hidden_states = hidden_states_cache[layer_index][memorizing_indices].to(device=device, dtype=torch.float64)

    reasoning_mean = reasoning_hidden_states.mean(dim=0)
    memorization_mean = memorizing_hidden_states.mean(dim=0)

    candidate_direction = reasoning_mean - memorization_mean
    magnitude = candidate_direction.norm(dim=-1, keepdim=True) + 1e-8  # Adding a small value to avoid division by zero
    candidate_direction = candidate_direction / magnitude  # Normalize the direction
    candidate_direction = candidate_direction.to(device=device, dtype=model.dtype)

    candidate_directions[layer_index] = candidate_direction
  
  return candidate_directions # Convert to the model's dtype

def get_device_map(model) -> dict[int, int]:
  device_map: dict[int, int] = {}

  match model.config.model_type:
    case name if name.startswith("huginn_"):
      if len(model.hf_device_map) == 1:
        # If the model is on a single device, assign all layers to that device
        device = model.hf_device_map['']
        for i in range(model.config.effective_expected_depth):
        # for i in range(model.config.effective_expected_depth + 1):
          device_map[i] = device
        return device_map

      # Prelude layers
      for i in range(model.config.n_layers_in_prelude):
        device_map[i] = model.hf_device_map["transformer.prelude"]

      # Core block layers (recurrent, may repeat device assignment)
      for i in range(
        model.config.mean_recurrence * model.config.n_layers_in_recurrent_block,
      ):
        device = model.hf_device_map[f"transformer.core_block.{i % model.config.n_layers_in_recurrent_block}"]
        device_map[model.config.n_layers_in_prelude + i] = device

      # Coda layers
      for i in range(model.config.n_layers_in_coda):
        device_map[model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block * model.config.mean_recurrence + i] = model.hf_device_map["transformer.coda"]

      # TODO: re-do the experiment with the final layer normalization
      # device_map[model.config.effective_expected_depth] = model.hf_device_map["transformer.ln_f"]

      return device_map
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")