from torch import nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from jaxtyping import Float
from typing import Literal

GradientMode = Literal["all_tokens", "last_token"]

def set_save_grad_hooks(
  model: nn.Module,
  gradient_mode: GradientMode,
):
  match model.config.model_type:
    case "huginn_raven":
      outputs = _save_grad_hooks_huginn(
        model=model,
        gradient_mode=gradient_mode,
      )
      return outputs
    case _:
      raise NotImplementedError(
        f"Model type {model.config.model_type} is not supported for saving hidden states gradients."
      )

def _register_full_backward_pre_hook(
  block: nn.Module,
  gradient_mode: GradientMode,
  depth_index: int,
  gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]],
):
  match gradient_mode:
    case "all_tokens":
      hook = block.register_full_backward_pre_hook(
        hook=lambda module, grad_output, depth_index=depth_index: gradients.update({
          depth_index: grad_output[0].detach().clone(),
        })
      )
    case "last_token":
      hook = block.register_full_backward_pre_hook(
        hook=lambda module, grad_output, depth_index=depth_index: gradients.update({
          depth_index: grad_output[0][:, -1:].detach().clone(),
        })
      )
    case _:
      raise ValueError(f"Unsupported gradient mode: {gradient_mode}")
  return hook

def _register_forward_hook(
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
        case "all_tokens":
          hidden_state.register_hook(
            lambda grad: gradients.update({
              # torch.Tensor.cpu() is important to avoid torch.OutOfMemoryError
              depth_index: grad.detach().clone(),
            })
          )
        case "last_token":
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
    
def _save_grad_hooks_huginn(
  model: nn.Module,
  gradient_mode: GradientMode,
):
  hooks: list[RemovableHandle] = []
  gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]] = {}

  # Registering save_grad hook for prelude blocks
  for block in model.transformer.prelude:
    print(f"Registering save_grad hook for prelude block with layer id {block.layer_id}")
    hook = _register_full_backward_pre_hook(
      block=block,
      gradient_mode=gradient_mode,
      depth_index=block.layer_id,
      gradients=gradients,
    )
    hooks.append(hook)

  # Registering save_grad hook for core blocks
  for block in model.transformer.core_block:
    print(f"Registering save_grad hook for core block with layer id {block.layer_id}")
    hook = _register_forward_hook(
      block=block,
      gradient_mode=gradient_mode,
      gradients=gradients,
    )
    hooks.append(hook)
  
  # Registering save_grad hook for coda blocks
  for block in model.transformer.coda:
    print(f"Registering save_grad hook for coda block with layer id {block.layer_id}")
    hook = _register_full_backward_pre_hook(
      block=block,
      gradient_mode=gradient_mode,
      depth_index=block.layer_id,
      gradients=gradients,
    )
    hooks.append(hook)

  # Registering save_grad hook for final layer normalization
  print("Registering save_grad hook for final layer normalization.")
  hook = _register_full_backward_pre_hook(
    block=model.transformer.ln_f,
    gradient_mode=gradient_mode,
    depth_index=model.config.effective_expected_depth,
    gradients=gradients,
  )
  hooks.append(hook)

  outputs = (hooks, gradients)

  return outputs