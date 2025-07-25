from utils.projection_hook import HookConfig
from utils.projection_hook import ProjectionHookMode
from utils.projection_hook import TokenModificationMode
from utils.projection_hook import DirectionNormalizationMode

from typing import TypedDict

class ProjectionHookConfig(TypedDict):
  """
  Reference:
  [1] https://github.com/FlyingPumba/steering-thinking-models/blob/0d5091a66b509504b6c61ccb4acf2650c9ac2408/utils/utils.py
  [2] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  steering_mode: ProjectionHookMode
  modification_mode: TokenModificationMode
  direction_normalization_mode: DirectionNormalizationMode
  layer_indices: list[int]
  hidden_states_hooks_config: HookConfig
  scale: float