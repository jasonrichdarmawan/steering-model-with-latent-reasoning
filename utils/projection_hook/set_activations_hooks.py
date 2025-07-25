from utils.projection_hook import ProjectionHookConfig
from utils.projection_hook import set_activations_hooks_huginn
from utils.projection_hook import set_activations_hooks_lirefs

from torch import nn
from torch import Tensor
from jaxtyping import Float
from torch.utils.hooks import RemovableHandle

def set_activations_hooks(
  model: nn.Module,
  feature_directions: dict[int, Float[Tensor, "n_embd"]],
  config: ProjectionHookConfig,
  overall_directions_magnitude: dict[int, Float[Tensor, ""]] | None = None,
  verbose: bool = True,
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
        feature_directions=feature_directions,
        config=config,
        overall_directions_magnitude=overall_directions_magnitude,
        hooks=hooks,
        verbose=verbose,
      )

      return hooks
    case "llama":
      hooks = set_activations_hooks_lirefs(
        model=model,
        feature_directions=feature_directions,
        config=config,
        overall_directions_magnitude=overall_directions_magnitude,
        hooks=hooks,
        verbose=verbose,
      )

      return hooks
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")