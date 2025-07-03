from jaxtyping import Float
from torch import Tensor

from typing import Literal

class ProjectionPostHookLiReFs:
  def __init__(
    self,
    type: Literal["mlp", "attention"],
    direction: Float[Tensor, "n_embd"],
    scale: float = 0.1,
  ):
    super().__init__()
    self.type = type
    self.direction = direction
    self.scale = scale
  
  def __call__(
    self,
    module, 
    input,
    output: tuple[Tensor, ...]
  ):
    match self.type:
      case "attention":
        projection = (output[0] @ self.direction).unsqueeze(-1) * self.direction
        output[0][:, :, :] = output[0] + self.scale * projection
      case "mlp":
        projection = (output @ self.direction).unsqueeze(-1) * self.direction
        output[:, :, :] = output + self.scale * projection