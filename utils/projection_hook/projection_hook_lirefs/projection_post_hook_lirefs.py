from utils.projection_hook import ProjectionHookMode
from utils.projection_hook import TokenModificationMode
from utils.projection_hook import DirectionNormalizationMode
from utils.projection_hook import compute_projection

from jaxtyping import Float
from torch import Tensor

from enum import Enum

class ProjectionPostHookLiReFsModuleType(Enum):
  HIDDEN_STATES = "HIDDEN_STATES" # Hidden states of the model
  ATTENTION = "ATTENTION" # Attention module
  MLP = "MLP" # Multilayer Perceptron

  def __str__(self):
    return self.value

class ProjectionPostHookLiReFs:
  def __init__(
    self,
    hook_type: ProjectionPostHookLiReFsModuleType,
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
    self.hook_type = hook_type
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
        if self.scale:
          raise ValueError("Scale should not be set for scale with overall magnitude normalization mode.")
        if self.overall_direction_magnitude is None:
          raise ValueError("Overall direction magnitude must be set for scale with overall magnitude normalization mode.")
  
  def __call__(
    self,
    module, 
    input,
    output: tuple[Tensor, ...]
  ):
    match self.hook_type:
      case ProjectionPostHookLiReFsModuleType.HIDDEN_STATES:
        match self.steering_mode:
          case ProjectionHookMode.FEATURE_AMPLIFICATION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                projection = compute_projection(
                  data=output[0][:, -1],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:, -1] += projection
              case TokenModificationMode.ALL_TOKENS:
                projection = compute_projection(
                  data=output[0],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:] += projection
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case ProjectionHookMode.FEATURE_ADDITION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                output[0][:, -1] += self.scale * self.feature_direction_normalized
              case TokenModificationMode.ALL_TOKENS:
                output[0][:] += self.scale * self.feature_direction_normalized
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case ProjectionHookMode.FEATURE_ABLATION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                projection = compute_projection(
                  data=output[0][:, -1],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:, -1] -= projection
              case TokenModificationMode.ALL_TOKENS:
                projection = compute_projection(
                  data=output[0],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:] -= projection
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case _:
            raise ValueError(f"Unsupported mode: {self.steering_mode}")
      case ProjectionPostHookLiReFsModuleType.ATTENTION:
        match self.steering_mode:
          case ProjectionHookMode.FEATURE_AMPLIFICATION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                projection = compute_projection(
                  data=output[0][:, -1],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:, -1] += projection
              case TokenModificationMode.ALL_TOKENS:
                projection = compute_projection(
                  data=output[0],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:] += projection
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case ProjectionHookMode.FEATURE_ADDITION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                output[0][:, -1] += self.scale * self.feature_direction_normalized
              case TokenModificationMode.ALL_TOKENS:
                output[0][:] += self.scale * self.feature_direction_normalized
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case ProjectionHookMode.FEATURE_ABLATION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                projection = compute_projection(
                  data=output[0][:, -1],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:, -1] -= projection
              case TokenModificationMode.ALL_TOKENS:
                projection = compute_projection(
                  data=output[0],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[0][:] -= projection
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case _:
            raise ValueError(f"Unsupported mode: {self.steering_mode}")
      case ProjectionPostHookLiReFsModuleType.MLP:
        match self.steering_mode:
          case ProjectionHookMode.FEATURE_AMPLIFICATION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                projection = compute_projection(
                  data=output[:, -1],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[:, -1] += projection
              case TokenModificationMode.ALL_TOKENS:
                projection = compute_projection(
                  data=output,
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[:] += projection
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case ProjectionHookMode.FEATURE_ADDITION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                output[:, -1] += self.scale * self.feature_direction_normalized
              case TokenModificationMode.ALL_TOKENS:
                output[:] += self.scale * self.feature_direction_normalized
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case ProjectionHookMode.FEATURE_ABLATION:
            match self.modification_mode:
              case TokenModificationMode.LAST_TOKEN:
                projection = compute_projection(
                  data=output[:, -1],
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[:, -1] -= projection
              case TokenModificationMode.ALL_TOKENS:
                projection = compute_projection(
                  data=output,
                  direction_normalization_mode=self.direction_normalization_mode,
                  feature_direction_normalized=self.feature_direction_normalized,
                  overall_direction_magnitude=self.overall_direction_magnitude,
                )
                output[:] -= projection
              case _:
                raise ValueError(f"Unsupported modification mode: {self.modification_mode}")
          case _:
            raise ValueError(f"Unsupported mode: {self.steering_mode}")
      case _:
        raise ValueError(f"Unsupported module type: {self.hook_type}")