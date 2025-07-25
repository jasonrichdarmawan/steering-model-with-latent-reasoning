from utils.projection_hook import ProjectionHookConfig
from utils.projection_hook.projection_hook_huginn import ProjectionPreHookHuginn
from utils.projection_hook.projection_hook_huginn import ProjectionPostHookHuginn

from torch import nn
from jaxtyping import Float
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from typing import NamedTuple

def set_activations_hooks_huginn(
  model: nn.Module,
  feature_directions: dict[int, Float[Tensor, "n_embd"]],
  config: ProjectionHookConfig,
  overall_directions_magnitude: dict[int, Float[Tensor, ""]] | None = None,
  hooks: list[RemovableHandle] | None = None,
) -> list[RemovableHandle]:
  if hooks is None:
    hooks = []

  core_block_last_layer_index: int = (
    model.config.n_layers_in_prelude 
    + model.config.n_layers_in_recurrent_block
  )
  coda_last_layer_index: int = (
    core_block_last_layer_index 
    + model.config.n_layers_in_coda
  )

  depth_indices_per_layer_index = _batch_depth_indices_per_layer_index(
    depth_indices=config["layer_indices"],
    n_layers_in_prelude=model.config.n_layers_in_prelude,
    n_layers_in_recurrent_block=model.config.n_layers_in_recurrent_block,
    mean_recurrence=model.config.mean_recurrence,
    n_layers_in_coda=model.config.n_layers_in_coda
  )

  for (layer_index, relative_layer_index), depth_indices in depth_indices_per_layer_index.items():
    module = None
    if layer_index < model.config.n_layers_in_prelude:
      module = model.transformer.prelude[relative_layer_index]
    elif layer_index < core_block_last_layer_index:
      module = model.transformer.core_block[relative_layer_index]
    elif layer_index < coda_last_layer_index:
      module = model.transformer.coda[relative_layer_index]
    elif layer_index == model.config.effective_expected_depth:
      module = model.transformer.ln_f
    else:
      raise ValueError(f"Module with layer index {layer_index} is out of bounds for the model.")

    if config["hidden_states_hooks_config"]["pre_hook"]:
      print(f"Registering pre-hook for module with layer index {layer_index}, relative index {relative_layer_index} and depth indices: {depth_indices}")
      pre_hook = ProjectionPreHookHuginn(
        steering_mode=config["steering_mode"],
        modification_mode=config["modification_mode"],
        direction_normalization_mode=config["direction_normalization_mode"],
        selected_depth_indices=depth_indices,
        feature_directions=feature_directions,
        overall_direction_magnitude=overall_directions_magnitude,
        scale=config["scale"],
      )
      hook = module.register_forward_pre_hook(
        hook=pre_hook,
        with_kwargs=True,
      )
      hooks.append(hook)

    if config["hidden_states_hooks_config"]["post_hook"]:
      print(f"Registering post-hook for module with layer index {layer_index}, relative index {relative_layer_index} and depth indices: {depth_indices}")
      post_hook = ProjectionPostHookHuginn(
        steering_mode=config["steering_mode"],
        modification_mode=config["modification_mode"],
        direction_normalization_mode=config["direction_normalization_mode"],
        selected_depth_indices=depth_indices,
        feature_directions=feature_directions,
        overall_direction_magnitude=overall_directions_magnitude,
        scale=config["scale"],
      )
      hook = module.register_forward_hook(
        hook=post_hook,
        with_kwargs=True,
      )
      hooks.append(hook)
  
  return hooks

class DepthIndex(NamedTuple):
  layer_index: int
  relative_layer_index: int

def _batch_depth_indices_per_layer_index(
  depth_indices: list[int],
  n_layers_in_prelude: int,
  n_layers_in_recurrent_block: int,
  mean_recurrence: int,
  n_layers_in_coda: int,
) -> dict[DepthIndex, list[int]]:
  """
  Batch depth indices by layer index.
  """
  recurrent_block_last_depth_index = (
    n_layers_in_prelude 
    + n_layers_in_recurrent_block 
    * mean_recurrence
  )
  coda_last_depth_index = (
    recurrent_block_last_depth_index 
    + n_layers_in_coda
  )

  coda_first_layer_index = (
    n_layers_in_prelude 
    + n_layers_in_recurrent_block
  )

  effective_expected_depth = (
    n_layers_in_prelude + (
      n_layers_in_recurrent_block 
      * mean_recurrence
    ) 
    + n_layers_in_coda
  )

  depth_indices_per_layer_index: dict[DepthIndex, list[int]] = {}
  for depth_index in depth_indices:
    # prelude
    if depth_index < n_layers_in_prelude:
      batch_index = DepthIndex(
        layer_index=depth_index,
        relative_layer_index=depth_index,
      )
    # core block
    elif depth_index < recurrent_block_last_depth_index:
      absolute_index = n_layers_in_prelude + (
        (depth_index - n_layers_in_prelude)
        % n_layers_in_recurrent_block
      )
      batch_index = DepthIndex(
        layer_index=absolute_index,
        relative_layer_index=absolute_index - n_layers_in_prelude,
      )
    # coda
    elif depth_index < coda_last_depth_index:
      absolute_index = coda_first_layer_index + (
        (depth_index - recurrent_block_last_depth_index)
        % n_layers_in_coda
      )
      batch_index = DepthIndex(
        layer_index=absolute_index,
        relative_layer_index=absolute_index - coda_first_layer_index,
      )
    elif depth_index == effective_expected_depth:
      batch_index = DepthIndex(
        layer_index=depth_index,
        relative_layer_index=0,
      )
    else:
      raise ValueError(f"Layer index {depth_index} is out of bounds for the model.")
    
    if batch_index not in depth_indices_per_layer_index:
      depth_indices_per_layer_index[batch_index] = []
  
    depth_indices_per_layer_index[batch_index].append(depth_index)

  return depth_indices_per_layer_index