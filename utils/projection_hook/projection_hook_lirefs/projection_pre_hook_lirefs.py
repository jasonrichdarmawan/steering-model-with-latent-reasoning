from utils.projection_hook import ProjectionHookMode
from utils.projection_hook import TokenModificationMode
from utils.projection_hook import DirectionNormalizationMode
from utils.projection_hook import compute_projection

from jaxtyping import Float
from torch import Tensor

class ProjectionPreHookLiReFs:
  def __init__(
    self,
    steering_mode: ProjectionHookMode,
    modification_mode: TokenModificationMode,
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
    self.modification_mode = modification_mode
    self.feature_direction_normalized = feature_direction / feature_direction.norm(dim=-1)
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
    match self.modification_mode:
      case TokenModificationMode.LAST_TOKEN:
        index = (slice(None), -1)
      case TokenModificationMode.ALL_TOKENS:
        index = (slice(None), slice(None))
      case _:
        raise ValueError("Unsupported TokenModificationMode: {self.modification_mode}")

    match self.steering_mode:
      case ProjectionHookMode.FEATURE_ADDITION:
        delta = self.scale * self.feature_direction_normalized
      case ProjectionHookMode.FEATURE_ABLATION | ProjectionHookMode.FEATURE_AMPLIFICATION:
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
        target_tensor = input[0] if isinstance(input, tuple) else input
        projection = compute_projection(
          data=target_tensor[index],
          direction_normalization_mode=self.direction_normalization_mode,
          feature_direction_normalized=self.feature_direction_normalized,
          overall_direction_magnitude=self.overall_direction_magnitude,
        )

        sign = 1 if self.steering_mode == ProjectionHookMode.FEATURE_AMPLIFICATION else -1
        delta = sign * projection
      case _:
        raise ValueError("Unsupported ProjectionHookMode: {self.steering_mode}")
      
    if isinstance(input, tuple):
      input[0][index] += delta
    else:
      input[index] += delta