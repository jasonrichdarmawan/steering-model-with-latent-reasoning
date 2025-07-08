from asepl_utils.save_grad_hooks import GradientMode

from torch import nn
from torch import Tensor
from jaxtyping import Float

def register_forward_hook_huginn(
  block: nn.Module,
  gradient_mode: GradientMode,
  gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]],
):
  def _forward_hook(
    hidden_state: Float[Tensor, "batch seq_len n_embd"],
    gradient_mode: GradientMode,
    depth_index: int,
    gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]],
  ):
    if hidden_state.requires_grad:
      match gradient_mode:
        case GradientMode.ALL_TOKENS:
          hidden_state.register_hook(
            lambda grad: gradients.update({
              depth_index: grad.detach().clone(),
            })
          )
        case GradientMode.LAST_TOKEN:
          hidden_state.register_hook(
            lambda grad: gradients.update({
              depth_index: grad[:, -1:].detach().clone(),
            })
          )
        case _:
          raise ValueError(f"Unsupported gradient mode: {gradient_mode}")

  hook = block.register_forward_hook(
    hook=lambda module, args, kwargs, output: _forward_hook(
      hidden_state=output[0],
      gradient_mode=gradient_mode,
      depth_index=kwargs["depth_idx"],
      gradients=gradients,
    ),
    with_kwargs=True,
  )

  return hook