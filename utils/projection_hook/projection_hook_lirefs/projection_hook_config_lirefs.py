from utils.projection_hook import ProjectionHookConfig
from utils.projection_hook import HookConfig

class ProjectionHookConfigLiReFs(ProjectionHookConfig):
  attention_hooks: HookConfig | None
  mlp_hooks: HookConfig | None