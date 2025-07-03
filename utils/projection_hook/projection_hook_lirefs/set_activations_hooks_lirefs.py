from utils.projection_hook.projection_hook_lirefs import ProjectionHookConfigLiReFs
from utils.projection_hook.projection_hook_lirefs import ProjectionPreHookLiReFs
from utils.projection_hook.projection_hook_lirefs import ProjectionPostHookLiReFs

from torch import nn
from torch import Tensor
from jaxtyping import Float
from torch.utils.hooks import RemovableHandle

def set_activations_hooks_lirefs(
  model: nn.Module,
  candidate_directions: Float[Tensor, "n_layers n_embd"],
  config: ProjectionHookConfigLiReFs,
  hooks: list[RemovableHandle] | None = None,
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

    direction = candidate_directions[layer_index]
    scale = config["scale"]

    # hidden_states
    if config["hidden_states_hooks"]:
      if config["hidden_states_hooks"]["pre_hook"]:
        pre_hook = ProjectionPreHookLiReFs(
          direction=direction,
          scale=scale,
        )
        hook = hidden_states[layer_index].register_forward_pre_hook(
          hook=pre_hook,
        )
        hooks.append(hook)
        print(f"Registering hidden_states pre-hook for {model.config.model_type} model at layer index {layer_index}")

      if config["hidden_states_hooks"]["post_hook"]:
        raise NotImplementedError(
          f"Post-hook for hidden states is not implemented for {model.config.model_type} model."
        )
    
    # attention
    if config["attention_hooks"]:
      if config["attention_hooks"]["pre_hook"]:
        raise NotImplementedError(
          f"Pre-hook for attention is not implemented for {model.config.model_type} model."
        )

      if config["attention_hooks"]["post_hook"]:
        post_hook = ProjectionPostHookLiReFs(
          type="attention",
          direction=direction,
          scale=scale,
        )
        hook = getattr(
          hidden_states[layer_index],
          attention_name,
        ).register_forward_hook(
          hook=post_hook,
        )
        hooks.append(hook)
        print(f"Registering attention post-hook for {model.config.model_type} model at layer index {layer_index}")

    # mlp
    if config["mlp_hooks"]:
      if config["mlp_hooks"]["pre_hook"]:
        raise NotImplementedError(
          f"Pre-hook for MLP is not implemented for {model.config.model_type} model."
        )

      if config["mlp_hooks"]["post_hook"]:
        post_hook = ProjectionPostHookLiReFs(
          type="mlp",
          direction=direction,
          scale=scale,
        )
        hook = getattr(
          hidden_states[layer_index],
          mlp_name,
        ).register_forward_hook(
          hook=post_hook,
        )
        hooks.append(hook)
        print(f"Registering MLP post-hook for {model.config.model_type} model at layer index {layer_index}")
  
  return hooks