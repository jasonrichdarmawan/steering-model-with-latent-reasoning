from utils.projection_hook import DirectionNormalizationMode

from torch import Tensor
from jaxtyping import Float

def compute_projection(
  data: Float[Tensor, "batch_size seq_len n_embd"],
  direction_normalization_mode: DirectionNormalizationMode,
  feature_direction_normalized: Float[Tensor, "n_embd"],
  overall_direction_magnitude: Float[Tensor, ""] | None = None,
):
  match direction_normalization_mode:
    case DirectionNormalizationMode.UNIT_VECTOR:
      return (data @ feature_direction_normalized).unsqueeze(-1) * feature_direction_normalized
    case DirectionNormalizationMode.SCALE_WITH_OVERALL_MAGNITUDE:
      return (data @ feature_direction_normalized).unsqueeze(-1) * feature_direction_normalized * overall_direction_magnitude
    case _:
      raise ValueError(f"Unsupported direction normalization mode: {direction_normalization_mode}")