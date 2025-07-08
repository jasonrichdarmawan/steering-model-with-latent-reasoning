from utils.projection_hook import ProjectionHookMode

from jaxtyping import Float
from torch import Tensor

class ProjectionPostHookHuginn:
  def __init__(
    self,
    mode: ProjectionHookMode,
    selected_depth_indices: list[int],
    directions: dict[int, Float[Tensor, "n_embd"]],
    scale: float = 1.0,
  ):
    super().__init__()
    self.mode = mode
    self.selected_depth_indices = selected_depth_indices
    self.candidate_directions = directions
    self.scale = scale

    match mode:
      case ProjectionHookMode.FEATURE_ADDITION:
        if scale is None:
          self.scale = 1.0
      case ProjectionHookMode.FEATURE_ABLATION:
        if scale:
          raise ValueError("Scale should not be set for ablation mode.")
      case _:
        raise ValueError(f"Unsupported mode: {mode}")
  
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
      match self.mode:
        case ProjectionHookMode.FEATURE_ADDITION:
          direction = self.candidate_directions[depth_index]
          output[0][:, :, :] = output[0] + (self.scale * direction)
        case ProjectionHookMode.FEATURE_ABLATION:
          direction = self.candidate_directions[depth_index]
          projection = (output[0] @ direction).unsqueeze(-1) * direction
          output[0][:, :, :] = output[0] - projection