from asepl_utils.save_grad_hooks import GradientMode
from asepl_utils.save_grad_hooks.save_grad_hooks_huginn import set_save_grad_hooks_huginn

from torch import nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from jaxtyping import Float

def set_save_grad_hooks(
  model: nn.Module,
  gradient_mode: GradientMode,
):
  match model.config.model_type:
    case "huginn_raven":
      outputs = set_save_grad_hooks_huginn(
        model=model,
        gradient_mode=gradient_mode,
      )
      return outputs
    case "llama":
      hooks: list[RemovableHandle] = []
      gradients: dict[int, Float[Tensor, "batch seq_len n_embd"]] = {}

      for layer_index, layer in enumerate(model.model.layers):
        print(f"Registering save_grad hook for layer {layer_index}")
        match gradient_mode:
          case GradientMode.LAST_TOKEN:
            hook = layer.register_full_backward_pre_hook(
              hook=lambda module, grad_output, layer_index=layer_index: gradients.update({
                layer_index: grad_output[0][:, -1:].detach().clone(),
              })
            )
          case GradientMode.ALL_TOKENS:
            hook = layer.register_full_backward_pre_hook(
              hook=lambda module, grad_output, layer_index=layer_index: gradients.update({
                layer_index: grad_output[0].detach().clone()
              })
            )
          case _:
            raise ValueError(f"Unsupported gradient mode: {gradient_mode}")
        hooks.append(hook)
      
      return hooks, gradients
    case _:
      raise NotImplementedError(
        f"Model type {model.config.model_type} is not supported for saving hidden states gradients."
      )