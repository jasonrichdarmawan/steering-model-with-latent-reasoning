from asepl_utils.save_grad_hooks import GradientMode
from asepl_utils.save_grad_hooks.save_grad_hooks_huginn import set_save_grad_hooks_huginn

from torch import nn

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
    case _:
      raise NotImplementedError(
        f"Model type {model.config.model_type} is not supported for saving hidden states gradients."
      )