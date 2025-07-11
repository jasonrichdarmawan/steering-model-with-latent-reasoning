from utils.projection_hook import ProjectionHookMode
from utils.projection_hook import DirectionNormalizationMode
from utils.projection_hook import compute_projection

from jaxtyping import Float
from torch import Tensor

class ProjectionPreHookLiReFs:
  def __init__(
    self,
    steering_mode: ProjectionHookMode,
    direction_normalization_mode: DirectionNormalizationMode,
    feature_direction: Float[Tensor, "n_embd"],
    overall_direction_magnitude: Float[Tensor, ""] | None = None,
    scale: float | None = None,
  ):
    """
    direction must be a unit vector, meaning its magnitude is 1
    """
    super().__init__()
    self.direction_normalization_mode = direction_normalization_mode
    self.steering_mode = steering_mode
    self.feature_direction = feature_direction
    self.feature_direction_normalized = self.feature_direction / self.feature_direction.norm(dim=-1)
    self.overall_direction_magnitude = overall_direction_magnitude
    self.scale = scale

    match steering_mode:
      case ProjectionHookMode.FEATURE_AMPLIFICATION:
        if scale:
          raise ValueError("Scale should not be set for feature amplification mode.")
      case ProjectionHookMode.FEATURE_ADDITION:
        if scale is None:
          raise ValueError("Scale must be set for feature addition mode.")
      case ProjectionHookMode.FEATURE_ABLATION:
        if scale:
          raise ValueError("Scale should not be set for feature addition or ablation mode.")
      case _:
        raise ValueError(f"Unsupported mode: {steering_mode}")
      
    match direction_normalization_mode:
      case DirectionNormalizationMode.UNIT_VECTOR:
        pass
      case DirectionNormalizationMode.SCALE_WITH_OVERALL_MAGNITUDE:
        if self.overall_direction_magnitude is None:
          raise ValueError("Overall direction magnitude must be set for scale with overall magnitude normalization mode.")
      case _:
        raise ValueError(f"Unsupported direction normalization mode: {direction_normalization_mode}")

  def __call__(
    self,
    module,
    input,
  ):
    match self.steering_mode:
      case ProjectionHookMode.FEATURE_ADDITION:
        input[0][:, :, :] = input[0] + (self.scale * self.feature_direction_normalized)
      case ProjectionHookMode.FEATURE_AMPLIFICATION:
        projection = compute_projection(
          data=input[0],
          direction_normalization_mode=self.direction_normalization_mode,
          feature_direction_normalized=self.feature_direction_normalized,
          overall_direction_magnitude=self.overall_direction_magnitude,
        )
        input[0][:, :, :] = input[0] + projection
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
        projection = compute_projection(
          data=input[0],
          direction_normalization_mode=self.direction_normalization_mode,
          feature_direction_normalized=self.feature_direction_normalized,
          overall_direction_magnitude=self.overall_direction_magnitude,
        )
        input[0][:, :, :] = input[0] - projection
      case _:
        raise ValueError(f"Unsupported mode: {self.mode}")