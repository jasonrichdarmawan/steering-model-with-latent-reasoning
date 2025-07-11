from enum import Enum

class DirectionNormalizationMode(Enum):
  """
  Enum for different methods of computing candidate directions.

  - UNIT_VECTOR: Computes the unit vector of the difference in mean between 
    the positive set (reasoning) and negative set (memorizing).

  - SCALE_WITH_OVERALL_MAGNITUDE: Scale the candidate direction with
    the magnitude of the combined positive (reasoning) set and negative set (memorizing).
    The purpose is to ensure that all candidate directions have a consistent magnitude,
    matching that of the mean activation vector of the combined positive and negative sets.
    This prevents features with naturally larger or smaller difference vectors from having a 
    disproportionate effect when they're used for steering the model's behavior later on. 
    It standardizes their "strength" while keeping their unique directional information.

  Notes:
  - Magnitude (or norm) is the length of a vector.
  - A unit vector is a vector whose mangitude (length) is exactly 1.
    Its purpose is to represent a direction

  Reference:
  [1] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/Intervention/utils.py#L247
  [2] https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/utils/utils.py#L267-L270 
  """
  UNIT_VECTOR = "UNIT_VECTOR"
  SCALE_WITH_OVERALL_MAGNITUDE = "SCALE_WITH_OVERALL_MAGNITUDE"

  def __str__(self):
    return self.value