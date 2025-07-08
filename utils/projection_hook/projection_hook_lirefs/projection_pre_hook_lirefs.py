from utils.projection_hook import ProjectionHookMode

from jaxtyping import Float
from torch import Tensor

class ProjectionPreHookLiReFs:
  def __init__(
    self,
    mode: ProjectionHookMode,
    direction: Float[Tensor, "n_embd"],
    scale: float | None = None,
  ):
    """
    direction must be a unit vector, meaning its magnitude is 1
    """
    super().__init__()
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
      
      # LiReFs specific modes
      case ProjectionHookMode.LIREFS_SOURCE_CODE:
        if scale is None:
          self.scale = 0.1
      case _:
        raise ValueError(f"Unsupported mode: {mode}")

  def __call__(
    self,
    module,
    input,
  ):
    match self.mode:
      case ProjectionHookMode.FEATURE_ADDITION:
        input[0][:, :, :] = input[0] + (self.scale * self.direction)
      case ProjectionHookMode.FEATURE_ABLATION:
        """
        we project the activation vector v 
        onto the direction vector d because 
        our goal is to isolate and modify
        the part of the activation that 
        corresponds to the concept 
        represented by d

        projecting v onto d (proj_d(v)) answer
        the question: "How much the concept
        d is present within the current 
        activation v?"
        The result is a vector that points
        in the same direction as d, with
        a magnitude representing the strength
        of that concept in v.

        by subtracting this projection
        (v - proj_d(v)), we are removing
        the component of the activation
        that aligns with out target concept,
        effectively "ablating" that feature
        from the model's processing stream.
        """
        projection = (input[0] @ self.direction).unsqueeze(-1) * self.direction
        input[0][:, :, :] = input[0] - projection
      
      # LiReFs specific modes
      case ProjectionHookMode.LIREFS_SOURCE_CODE:
        projection = (input[0] @ self.direction).unsqueeze(-1) * self.direction
        input[0][:, :, :] = input[0] + (self.scale * projection)
      case _:
        raise ValueError(f"Unsupported mode: {self.mode}")