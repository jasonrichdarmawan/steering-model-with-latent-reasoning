from utils.projection_hook import ProjectionHookConfig
from utils.projection_hook import HookConfig

class ProjectionHookConfigLiReFs(ProjectionHookConfig):
  attention_hooks_config: HookConfig | None
  mlp_hooks_config: HookConfig | None