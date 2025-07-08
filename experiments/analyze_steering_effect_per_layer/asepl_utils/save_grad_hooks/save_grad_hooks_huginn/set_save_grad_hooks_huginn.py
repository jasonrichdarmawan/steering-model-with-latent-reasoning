from asepl_utils.save_grad_hooks import GradientMode
from asepl_utils.save_grad_hooks.save_grad_hooks_huginn import register_full_backward_pre_hook_huginn
from asepl_utils.save_grad_hooks.save_grad_hooks_huginn import register_forward_hook_huginn

from torch import nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from jaxtyping import Float

def set_save_grad_hooks_huginn(
  model: nn.Module,
  gradient_mode: GradientMode,
):
  hooks: list[RemovableHandle] = []
  gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]] = {}

  # Registering save_grad hook for prelude blocks
  for block in model.transformer.prelude:
    print(f"Registering save_grad hook for prelude block with layer id {block.layer_id}")
    hook = register_full_backward_pre_hook_huginn(
      block=block,
      gradient_mode=gradient_mode,
      depth_index=block.layer_id,
      gradients=gradients,
    )
    hooks.append(hook)

  # Registering save_grad hook for core blocks
  for block in model.transformer.core_block:
    print(f"Registering save_grad hook for core block with layer id {block.layer_id}")
    hook = register_forward_hook_huginn(
      block=block,
      gradient_mode=gradient_mode,
      gradients=gradients,
    )
    hooks.append(hook)
  
  # Registering save_grad hook for coda blocks
  for block in model.transformer.coda:
    print(f"Registering save_grad hook for coda block with layer id {block.layer_id}")
    hook = register_full_backward_pre_hook_huginn(
      block=block,
      gradient_mode=gradient_mode,
      depth_index=block.layer_id,
      gradients=gradients,
    )
    hooks.append(hook)

  # Registering save_grad hook for final layer normalization
  print("Registering save_grad hook for final layer normalization.")
  hook = register_full_backward_pre_hook_huginn(
    block=model.transformer.ln_f,
    gradient_mode=gradient_mode,
    depth_index=model.config.effective_expected_depth,
    gradients=gradients,
  )
  hooks.append(hook)

  outputs = (hooks, gradients)

  return outputs