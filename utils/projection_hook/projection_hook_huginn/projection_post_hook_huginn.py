from jaxtyping import Float
from torch import Tensor

class ProjectionPostHookHuginn:
  def __init__(
    self,
    selected_depth_indices: list[int],
    candidate_directions: Float[Tensor, "n_layers n_embd"],
    scale: float = 0.1
  ):
    super().__init__()
    self.selected_depth_indices = selected_depth_indices
    self.candidate_directions = candidate_directions
    self.scale = scale
  
  def __call__(
    self,
    module, 
    args,
    kwargs,
    output: tuple[Tensor, ...]
  ):
    depth_index: int = kwargs["depth_idx"]
    if ( 
      (depth_index in module.depth_indices)
      and depth_index in self.selected_depth_indices
    ):
      direction = self.candidate_directions[depth_index]
      projection = (output[0] @ direction).unsqueeze(-1) * direction
      output[0][:, :, :] = output[0] + self.scale * projection