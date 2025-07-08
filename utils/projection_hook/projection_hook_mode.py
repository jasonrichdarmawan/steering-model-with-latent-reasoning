from enum import Enum

class ProjectionHookMode(Enum):
  """
  - LIREFS_SOURCE_CODE: It doesn't make sense to calculate 
  the projection for feature addition. Suppose the 
  concept direction is not present in the activation, 
  the projection will be zero, and the addition will not 
  change the activation.

  In addition, the reference's implementation is 
  different from the paper.
  The paper adds the scaled direction vector directly 
  to the output.

  Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/Intervention/utils.py#L253-L256
  """
  FEATURE_ADDITION = "FEATURE_ADDITION"
  FEATURE_ABLATION = "FEATURE_ABLATION"
  LIREFS_SOURCE_CODE = "LIREFS_SOURCE_CODE"

  def __str__(self):
    return self.value