from torch import nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from jaxtyping import Float

def set_save_grad_hooks(
  model: nn.Module,
):
  match model.config.model_type:
    case name if name.startswith("huginn_"):
      outputs = _save_grad_hooks_huginn(
        model=model,
      )
      return outputs
    case _:
      raise NotImplementedError(
        f"Model type {model.config.model_type} is not supported for saving hidden states gradients."
      )
    
def _save_grad_hooks_huginn(
  model: nn.Module,
):
  hooks: list[RemovableHandle] = []
  gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]] = {}

  # Registering save_grad hook for prelude blocks
  for block in model.transformer.prelude:
    print(f"Registering save_grad hook for prelude block with layer id {block.layer_id}")
    hook = block.register_full_backward_pre_hook(
      hook=lambda module, grad_output, layer_index=block.layer_id: gradients.update({
        layer_index: grad_output[0],
      }),
    )
    hooks.append(hook)

  # Registering save_grad hook for core blocks
  def forward_hook(
    module: nn.Module,
    args,
    kwargs,
    output: Float[Tensor, "batch seq_len n_embd"],
  ):
    depth_index: int = kwargs["depth_idx"]
    hidden_state = output[0]
    if hidden_state.requires_grad:
      hidden_state.register_hook(
        lambda grad: gradients.update({
          depth_index: grad,
        })
      )
  for block in model.transformer.core_block:
    print(f"Registering save_grad hook for core block with layer id {block.layer_id}")
    block: nn.Module
    hook = block.register_forward_hook(
      hook=forward_hook,
      with_kwargs=True,
    )
    hooks.append(hook)
  
  # Registering save_grad hook for coda blocks
  for block in model.transformer.coda:
    print(f"Registering save_grad hook for coda block with layer id {block.layer_id}")
    hook = block.register_full_backward_pre_hook(
      hook=lambda module, grad_output, depth_index=block.layer_id: gradients.update({
        depth_index: grad_output[0],
      }),
    )
    hooks.append(hook)

  # Registering save_grad hook for final layer normalization
  print("Registering save_grad hook for final layer normalization.")
  hook = model.transformer.ln_f.register_full_backward_pre_hook(
    hook=lambda module, grad_output, depth_index=model.config.effective_expected_depth: gradients.update({
      depth_index: grad_output[0],
    }),
  )
  hooks.append(hook)

  outputs = (hooks, gradients)

  return outputs