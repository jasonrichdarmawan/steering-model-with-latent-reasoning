from utils.projection_hook import ProjectionHookConfig
from utils.projection_hook import set_activations_hooks_huginn
from utils.projection_hook import set_activations_hooks_lirefs

from torch import nn
from torch import Tensor
from jaxtyping import Float
from torch.utils.hooks import RemovableHandle

def set_activations_hooks(
  model: nn.Module,
  candidate_directions: Float[Tensor, "n_layers n_embd"],
  config: ProjectionHookConfig,
):
  """
  Note:
  Unlike LiReFs [1], which selects a candidate direction from a specific layer and applies it to the activations of every layer in the model, this implementation applies each candidate direction only to its corresponding layer as specified in `layer_indices`. This allows for more targeted interventions at specific layers rather than a global modification across all layers.

  Reference: 
  [1] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  hooks: list[RemovableHandle] = []

  match model.config.model_type:
    case "huginn_raven":
      hooks = set_activations_hooks_huginn(
        model=model,
        candidate_directions=candidate_directions,
        config=config,
        hooks=hooks
      )

      return hooks
    case "llama":
      hooks = set_activations_hooks_lirefs(
        model=model,
        candidate_directions=candidate_directions,
        config=config,
        hooks=hooks
      )

      return hooks
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")