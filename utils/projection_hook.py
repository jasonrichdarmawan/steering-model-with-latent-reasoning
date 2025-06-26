from typing import TypedDict
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

class ProjectionHookConfig(TypedDict):
  """
  Reference:
  [1] https://github.com/FlyingPumba/steering-thinking-models/blob/0d5091a66b509504b6c61ccb4acf2650c9ac2408/utils/utils.py
  [2] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  layer_indices: list[int]
  candidate_directions: list[Float[Tensor, "n_layers n_embd"]]
  pre_hook: bool
  post_hook: bool
  scale: float

def set_activations_hooks(
  model: nn.Module,
  candidate_directions: Float[Tensor, "n_layers n_embd"],
  config: ProjectionHookConfig,
):
  """
  Note:
  Unlike LiReFs [1], which selects a candidate direction from a specific layer and applies it to the activations of every layer in the model, this implementation applies each candidate direction only to its corresponding layer as specified in `layer_indices`. This allows for more targeted interventions at specific layers rather than a global modification across all layers.

  Reference: 
  [1] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  hooks: list[RemovableHandle] = []

  match model.config.model_type:
    case name if name.startswith("huginn_"):
      hooks = _set_activations_hooks_huginn(
        model=model,
        candidate_directions=candidate_directions,
        config=config,
        hooks=hooks
      )
  
  return hooks

class ProjectionPreHook:
  def __init__(
    self,
    selected_depth_indices: list[int],
    candidate_directions: Float[Tensor, "n_layers n_embd"], 
    scale: float = 0.1
  ):
    super().__init__()
    self.selected_depth_indices = selected_depth_indices
    self.candidate_directions = candidate_directions
    self.scale = scale

  def __call__(
    self,
    module: nn.Module, 
    args,
    kwargs,
  ):
    depth_index: int = kwargs["depth_idx"]
    if ( 
      depth_index in module.depth_indices
      and depth_index in self.selected_depth_indices
    ):
      direction = self.candidate_directions[depth_index]
      projection = (kwargs["x"] @ direction).unsqueeze(-1) * direction  # (batch, sequence_length, 1)
      kwargs["x"][:, :, :] = kwargs["x"] + self.scale * projection

class ProjectionPostHook:
  def __init__(
    self,
    selected_depth_indices: list[int],
    candidate_directions: Float[Tensor, "n_layers n_embd"],
    scale: float = 0.1
  ):
    super().__init__()
    self.selected_depth_indices = selected_depth_indices
    self.candidate_directions = candidate_directions
    self.scale = scale
  
  def __call__(
    self,
    module, 
    args,
    kwargs,
    output: tuple[Tensor, ...]
  ):
    depth_index: int = kwargs["depth_idx"]
    if ( 
      (depth_index in module.depth_indices)
      and depth_index in self.selected_depth_indices
    ):
      direction = self.candidate_directions[depth_index]
      projection = (output[0] @ direction).unsqueeze(-1) * direction # (batch, sequence_length, 1)
      output[0][:, :, :] = output[0] + self.scale * projection

def _set_activations_hooks_huginn(
  model: nn.Module,
  candidate_directions: Float[Tensor, "n_layers n_embd"],
  config: ProjectionHookConfig,
  hooks: list[RemovableHandle] | None = None
) -> list[RemovableHandle]:
  if hooks is None:
    hooks = []

  core_block_last_layer_index: int = model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block
  coda_last_layer_index: int = core_block_last_layer_index + model.config.n_layers_in_coda

  depth_indices_batched: dict[
    tuple[int, int], list[int]
  ] = _batch_depths_by_recurrence(
    depth_indices=config["layer_indices"],
    n_layers_in_prelude=model.config.n_layers_in_prelude,
    n_layers_in_recurrent_block=model.config.n_layers_in_recurrent_block,
    mean_recurrence=model.config.mean_recurrence,
    n_layers_in_coda=model.config.n_layers_in_coda
  )

  for layer_indices, depth_indices in depth_indices_batched.items():
    module = None
    if layer_indices[0] < model.config.n_layers_in_prelude:
      module = model.transformer.prelude[layer_indices[1]]
    elif layer_indices[0] < core_block_last_layer_index:
      module = model.transformer.core_block[layer_indices[1]]
    elif layer_indices[0] < coda_last_layer_index:
      module = model.transformer.coda[layer_indices[1]]
    # TODO: re-do the experiment with the final layer normalization
    # elif layer_indices[0] == model.config.effective_expected_depth:
    #   module = model.transformer.ln_f
    else:
      raise ValueError(f"Module with layer index {layer_indices[0]} is out of bounds for the model.")

    if config["pre_hook"]:
      print(f"Registering pre-hook for module with layer indices: absolute index {layer_indices[0]}, relative index {layer_indices[1]} and depth indices: {depth_indices}")
      pre_hook = ProjectionPreHook(
        selected_depth_indices=depth_indices,
        candidate_directions=candidate_directions,
        scale=config["scale"]
      )
      hook = module.register_forward_pre_hook(
        hook=pre_hook,
        with_kwargs=True,
      )
      hooks.append(hook)

    if config["post_hook"]:
      print(f"Registering post-hook for module with absolute index {layer_indices[0]}, relative index {layer_indices[1]} and depth indices: {depth_indices}")
      post_hook = ProjectionPostHook(
        selected_depth_indices=depth_indices,
        candidate_directions=candidate_directions,
        scale=config["scale"]
      )
      hook = module.register_forward_hook(
        hook=post_hook,
        with_kwargs=True,
      )
      hooks.append(hook)
  
  return hooks

def _batch_depths_by_recurrence(
  depth_indices: list[int],
  n_layers_in_prelude: int,
  n_layers_in_recurrent_block: int,
  mean_recurrence: int,
  n_layers_in_coda: int,
) -> dict[tuple[int, int], list[int]]:
  """
  Batch layer indices by recurrence.
  """
  core_block_last_depth_index = n_layers_in_prelude + n_layers_in_recurrent_block * mean_recurrence
  coda_last_depth_index = core_block_last_depth_index + n_layers_in_coda

  coda_first_layer_index = n_layers_in_prelude + n_layers_in_recurrent_block

  # effective_expected_depth = n_layers_in_prelude + (
  #   n_layers_in_recurrent_block * mean_recurrence
  # ) + n_layers_in_coda

  layer_indices_batched: dict[tuple[int, int], list[int]] = {}
  for depth_index in depth_indices:
    # prelude
    if depth_index < n_layers_in_prelude:
      batch_index = (depth_index, depth_index)
    # core block
    elif depth_index < core_block_last_depth_index:
      absolute_index = n_layers_in_prelude + (
        (depth_index - n_layers_in_prelude)
        % n_layers_in_recurrent_block
      )
      batch_index = (
        absolute_index, 
        absolute_index - n_layers_in_prelude # relative index
      )
    # coda
    elif depth_index < coda_last_depth_index:
      absolute_index = coda_first_layer_index + (
        (depth_index - core_block_last_depth_index)
        % n_layers_in_coda
      )
      batch_index = (
        absolute_index, 
        absolute_index - coda_first_layer_index # relative index
      )
    # ln_f
    # TODO: re-do the experiment with the final layer normalization
    # elif depth_index == effective_expected_depth:
    #   batch_index = (depth_index, 0)
    else:
      raise ValueError(f"Layer index {depth_index} is out of bounds for the model.")
    
    if batch_index not in layer_indices_batched:
      layer_indices_batched[batch_index] = []
  
    layer_indices_batched[batch_index].append(depth_index)

  return layer_indices_batched