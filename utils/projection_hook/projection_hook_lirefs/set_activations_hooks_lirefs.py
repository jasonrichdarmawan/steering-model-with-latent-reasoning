from utils.projection_hook.projection_hook_lirefs import ProjectionHookConfigLiReFs
from utils.projection_hook.projection_hook_lirefs import ProjectionPreHookLiReFs
from utils.projection_hook.projection_hook_lirefs import ProjectionPostHookLiReFs
from utils.projection_hook.projection_hook_lirefs import ProjectionPostHookLiReFsModuleType

from torch import nn
from torch import Tensor
from jaxtyping import Float
from torch.utils.hooks import RemovableHandle

def set_activations_hooks_lirefs(
  model: nn.Module,
  feature_directions: dict[int, Float[Tensor, "n_embd"]],
  config: ProjectionHookConfigLiReFs,
  overall_directions_magnitude: dict[int, Float[Tensor, ""]] | None = None,
  hooks: list[RemovableHandle] | None = None,
  verbose: bool = True,
) -> list[RemovableHandle]:
  """
  Set activation hooks for Llama models.
  """
  if hooks is None:
    hooks = []

  match model.config.model_type:
    case "llama":
      hidden_states = model.model.layers
      attention_name = "self_attn"
      mlp_name = "mlp"
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")

  for layer_index in config["layer_indices"]:

    feature_direction = feature_directions[layer_index]
    overall_direction_magnitude = overall_directions_magnitude[layer_index] if overall_directions_magnitude else None
    scale = config["scale"]

    # hidden_states
    if config["hidden_states_hooks_config"]:
      if config["hidden_states_hooks_config"]["pre_hook"]:
        if verbose:
          print(f"Registering hidden_states pre-hook for {model.config.model_type} model at layer index {layer_index}")
        pre_hook = ProjectionPreHookLiReFs(
          steering_mode=config["steering_mode"],
          modification_mode=config["modification_mode"],
          direction_normalization_mode=config["direction_normalization_mode"],
          feature_direction=feature_direction,
          overall_direction_magnitude=overall_direction_magnitude,
          scale=scale,
        )
        hook = hidden_states[layer_index].register_forward_pre_hook(
          hook=pre_hook,
        )
        hooks.append(hook)

      if config["hidden_states_hooks_config"]["post_hook"]:
        if verbose:
          print(f"Registering hidden_states post-hook for {model.config.model_type} model at layer index {layer_index}")
        post_hook = ProjectionPostHookLiReFs(
          hook_type=ProjectionPostHookLiReFsModuleType.HIDDEN_STATES,
          steering_mode=config["steering_mode"],
          modification_mode=config["modification_mode"],
          direction_normalization_mode=config["direction_normalization_mode"],
          feature_direction=feature_direction,
          overall_direction_magnitude=overall_direction_magnitude,
          scale=scale,
        )
        hook = hidden_states[layer_index].register_forward_hook(
          hook=post_hook,
        )
        hooks.append(hook)
    
    # attention
    if config["attention_hooks_config"]:
      if config["attention_hooks_config"]["pre_hook"]:
        raise NotImplementedError(
          f"Pre-hook for attention is not implemented for {model.config.model_type} model."
        )

      if config["attention_hooks_config"]["post_hook"]:
        if verbose:
          print(f"Registering attention post-hook for {model.config.model_type} model at layer index {layer_index}")
        post_hook = ProjectionPostHookLiReFs(
          hook_type=ProjectionPostHookLiReFsModuleType.ATTENTION,
          steering_mode=config["steering_mode"],
          modification_mode=config["modification_mode"],
          direction_normalization_mode=config["direction_normalization_mode"],
          feature_direction=feature_direction,
          overall_direction_magnitude=overall_direction_magnitude,
          scale=scale,
        )
        hook = getattr(
          hidden_states[layer_index],
          attention_name,
        ).register_forward_hook(
          hook=post_hook,
        )
        hooks.append(hook)

    # mlp
    if config["mlp_hooks_config"]:
      if config["mlp_hooks_config"]["pre_hook"]:
        raise NotImplementedError(
          f"Pre-hook for MLP is not implemented for {model.config.model_type} model."
        )

      if config["mlp_hooks_config"]["post_hook"]:
        if verbose:
          print(f"Registering MLP post-hook for {model.config.model_type} model at layer index {layer_index}")
        post_hook = ProjectionPostHookLiReFs(
          hook_type=ProjectionPostHookLiReFsModuleType.MLP,
          steering_mode=config["steering_mode"],
          modification_mode=config["modification_mode"],
          direction_normalization_mode=config["direction_normalization_mode"],
          feature_direction=feature_direction,
          overall_direction_magnitude=overall_direction_magnitude,
          scale=scale,
        )
        hook = getattr(
          hidden_states[layer_index],
          mlp_name,
        ).register_forward_hook(
          hook=post_hook,
        )
        hooks.append(hook)
  
  return hooks