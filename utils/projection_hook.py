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
      core_block_last_index: int = model.config.n_layers_in_prelude + model.config.n_layers_in_recurrent_block * model.config.mean_recurrence
      coda_last_index: int = core_block_last_index + model.config.n_layers_in_coda

      for layer_index in config["layer_indices"]:
        module = None
        if layer_index < model.config.n_layers_in_prelude:
          module = model.transformer.prelude[layer_index]
        elif layer_index < core_block_last_index:
          module = model.transformer.core_block[layer_index - core_block_last_index]
        elif layer_index < coda_last_index:
          module = model.transformer.coda[layer_index - coda_last_index]
        else:
          raise ValueError(f"Layer index {layer_index} is out of bounds for the model.")

        if config["pre_hook"]:
          pre_hook = ProjectionPreHook(
            direction=candidate_directions[layer_index],
            scale=config["scale"]
          )
          hook = module.register_forward_pre_hook(
            pre_hook
          )
          hooks.append(hook)

        if config["post_hook"]:
          post_hook = ProjectionPostHook(
            direction=candidate_directions[layer_index],
            scale=config["scale"]
          )
          hook = module.register_forward_hook(
            post_hook
          )
          hooks.append(hook)
  
  return hooks

def remove_hooks(hooks: list[RemovableHandle]):
  for hook in hooks:
    hook.remove()

class ProjectionPreHook:
  def __init__(
    self, 
    direction: Float[Tensor, "n_embd"], 
    scale: float = 0.1
  ):
    super().__init__()
    self.direction = direction
    self.scale = scale

  def __call__(
    self,
    module, 
    input: tuple[Tensor, ...]
  ):
    projection = (input[0] @ self.direction).unsqueeze(-1) * self.direction  # (batch, sequence_length, 1)
    input[0][:, :, :] = input[0] + self.scale * projection

class ProjectionPostHook:
  def __init__(
    self,
    direction: Float[Tensor, "n_embd"],
    scale: float = 0.1
  ):
    super().__init__()
    self.direction = direction
    self.scale = scale
  
  def __call__(
    self,
    module, 
    input: tuple[Tensor, ...],
    output: tuple[Tensor, ...]
  ):
    projection = (output[0] @ self.direction).unsqueeze(-1) * self.direction # (batch, sequence_length, 1)
    output[0][:, :, :] = output[0] + self.scale * projection