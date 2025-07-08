from asepl_utils.save_grad_hooks import GradientMode

from torch import nn
from jaxtyping import Float
from torch import Tensor

def register_full_backward_pre_hook_huginn(
  block: nn.Module,
  gradient_mode: GradientMode,
  depth_index: int,
  gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]],
):
  match gradient_mode:
    case GradientMode.ALL_TOKENS:
      hook = block.register_full_backward_pre_hook(
        hook=lambda module, grad_output, depth_index=depth_index: gradients.update({
          depth_index: grad_output[0].detach().clone(),
        })
      )
    case GradientMode.LAST_TOKEN:
      hook = block.register_full_backward_pre_hook(
        hook=lambda module, grad_output, depth_index=depth_index: gradients.update({
          depth_index: grad_output[0][:, -1:].detach().clone(),
        })
      )
    case _:
      raise ValueError(f"Unsupported gradient mode: {gradient_mode}")
  return hook