from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

def set_activations_hooks(
  model: nn.Module,
  layer_name: str,
  direction: Float[Tensor, "n_embd"],
  scale: float = 0.1
):
  hooks: list[RemovableHandle] = []

  match model.config.model_type:
    case name if name.startswith("huginn_"):
      module: nn.Module = model.transformer.prelude[0]
      
      pre_hook = ProjectionPreHook(
        direction=direction,
        scale=scale
      )
      hook = module.register_forward_pre_hook(
        pre_hook
      )
      hooks.append(hook)

      post_hook = ProjectionPostHook(
        direction=direction,
        scale=scale
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