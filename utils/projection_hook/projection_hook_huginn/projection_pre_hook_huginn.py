from utils.projection_hook import ProjectionHookMode
from utils.projection_hook import TokenModificationMode
from utils.projection_hook import DirectionNormalizationMode
from utils.projection_hook import compute_projection

from jaxtyping import Float
from torch import Tensor

class ProjectionPreHookHuginn:
  def __init__(
    self,
    steering_mode: ProjectionHookMode,
    modification_mode: TokenModificationMode,
    direction_normalization_mode: DirectionNormalizationMode,
    selected_depth_indices: list[int],
    feature_directions: dict[int, Float[Tensor, "n_embd"]],
    overall_direction_magnitude: dict[int, Float[Tensor, "n_embd"]] | None = None,
    scale: float | None = None,
  ):
    """
    candidate_directions muust be a list of unit vectors, meaning their magnitudes are 1.
    """
    super().__init__()
    self.steering_mode = steering_mode
    self.modification_mode = modification_mode
    self.selected_depth_indices = selected_depth_indices
    self.feature_directions_normalized = {
      depth_index: direction / direction.norm(dim=-1)
      for depth_index, direction in feature_directions.items()
    }
    self.direction_normalization_mode = direction_normalization_mode
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
          raise ValueError("Scale should not be set for ablation mode.")
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
      case _:
        raise ValueError(f"Unsupported direction normalization mode: {direction_normalization_mode}")

  def __call__(
    self,
    module, 
    args,
    kwargs,
  ):
    depth_index: int = kwargs["depth_idx"]
    if ( 
      depth_index in module.depth_indices
      and depth_index in self.selected_depth_indices
    ):
      match self.steering_mode:
        case ProjectionHookMode.FEATURE_AMPLIFICATION:
          feature_direction_normalized = self.feature_directions_normalized[depth_index]
          overall_direction_magnitude = (
            self.overall_direction_magnitude[depth_index] 
            if self.overall_direction_magnitude 
            else None
          )
          match self.modification_mode:
            case TokenModificationMode.LAST_TOKEN:
              projection = compute_projection(
                data=kwargs["x"][:, -1],
                direction_normalization_mode=self.direction_normalization_mode,
                feature_direction_normalized=feature_direction_normalized,
                overall_direction_magnitude=overall_direction_magnitude,
              )
              kwargs["x"][:, -1] += projection
            case TokenModificationMode.ALL_TOKENS:
              projection = compute_projection(
                data=kwargs["x"],
                direction_normalization_mode=self.direction_normalization_mode,
                feature_direction_normalized=feature_direction_normalized,
                overall_direction_magnitude=overall_direction_magnitude,
              )
              kwargs["x"] += projection
            case _:
              raise ValueError(f"Unsupported token modification mode: {self.matching_mode}")
        case ProjectionHookMode.FEATURE_ADDITION:
          direction = self.feature_directions_normalized[depth_index]
          match self.modification_mode:
            case TokenModificationMode.LAST_TOKEN:
              kwargs["x"][:, -1] += self.scale * direction
            case TokenModificationMode.ALL_TOKENS:
              kwargs["x"] += self.scale * direction
            case _:
              raise ValueError(f"Unsupported token modification mode: {self.matching_mode}")
        case ProjectionHookMode.FEATURE_ABLATION:
          feature_direction_normalized = self.feature_directions_normalized[depth_index]
          overall_direction_magnitude = (
            self.overall_direction_magnitude[depth_index] 
            if self.overall_direction_magnitude 
            else None
          )
          match self.modification_mode:
            case TokenModificationMode.LAST_TOKEN:
              projection = compute_projection(
                data=kwargs["x"][:, -1],
                direction_normalization_mode=self.direction_normalization_mode,
                feature_direction_normalized=feature_direction_normalized,
                overall_direction_magnitude=overall_direction_magnitude,
              )
              kwargs["x"][:, -1] -= projection
            case TokenModificationMode.ALL_TOKENS:
              projection = compute_projection(
                data=kwargs["x"],
                direction_normalization_mode=self.direction_normalization_mode,
                feature_direction_normalized=feature_direction_normalized,
                overall_direction_magnitude=overall_direction_magnitude,
              )
              kwargs["x"] -= projection
            case _:
              raise ValueError(f"Unsupported token modification mode: {self.matching_mode}")
        case _:
          raise ValueError(f"Unsupported steering mode: {self.steering_mode}")