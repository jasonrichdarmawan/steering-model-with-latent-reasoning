from utils.projection_hook import ProjectionHookMode

from jaxtyping import Float
from torch import Tensor

from enum import Enum

class ProjectionPostHookLiReFsModuleType(Enum):
  ATTENTION = "ATTENTION" # Attention module
  MLP = "MLP" # Multilayer Perceptron

class ProjectionPostHookLiReFs:
  def __init__(
    self,
    type: ProjectionPostHookLiReFsModuleType,
    mode: ProjectionHookMode,
    direction: Float[Tensor, "n_embd"],
    scale: float | None = None,
  ):
    """
    direction must be a unit vector, meaning its magnitude is 1
    """
    super().__init__()
    self.type = type
    self.mode = mode
    self.direction = direction
    self.scale = scale

    match mode:
      case ProjectionHookMode.FEATURE_ADDITION:
        if scale is None:
          self.scale = 1.0
      case ProjectionHookMode.FEATURE_ABLATION:
        if scale:
          raise ValueError("Scale should not be set for feature addition or ablation mode.")
      case ProjectionHookMode.LIREFS_SOURCE_CODE:
        if scale is None:
          self.scale = 0.1
      case _:
        raise ValueError(f"Unsupported mode: {mode}")

  
  def __call__(
    self,
    module, 
    input,
    output: tuple[Tensor, ...]
  ):
    match self.type:
      case ProjectionPostHookLiReFsModuleType.ATTENTION:
        match self.mode:
          case ProjectionHookMode.FEATURE_ADDITION:
            output[0][:, :, :] = output[0] + (self.scale * self.direction)
          case ProjectionHookMode.FEATURE_ABLATION:
            projection = (output[0] @ self.direction).unsqueeze(-1) * self.direction
            output[0][:, :, :] = output[0] - projection
          
          # LiReFs specific modes
          case ProjectionHookMode.LIREFS_SOURCE_CODE:
            projection = (output[0] @ self.direction).unsqueeze(-1) * self.direction
            output[0][:, :, :] = output[0] + (self.scale * projection)
          case _:
            raise ValueError(f"Unsupported mode: {self.mode}")
      case ProjectionPostHookLiReFsModuleType.MLP:
        match self.mode:
          case ProjectionHookMode.FEATURE_ADDITION:
            output[:, :, :] = output + (self.scale * self.direction)
          case ProjectionHookMode.FEATURE_ABLATION:
            projection = (output @ self.direction).unsqueeze(-1) * self.direction
            output[:, :, :] = output - projection
          
          # LiReFs specific modes
          case ProjectionHookMode.LIREFS_SOURCE_CODE:
            projection = (output @ self.direction).unsqueeze(-1) * self.direction
            output[:, :, :] = output + (self.scale * projection)
          case _:
            raise ValueError(f"Unsupported mode: {self.mode}")
      case _:
        raise ValueError(f"Unsupported module type: {self.type}")