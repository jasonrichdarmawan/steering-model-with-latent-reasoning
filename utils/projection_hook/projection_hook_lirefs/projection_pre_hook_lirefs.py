from jaxtyping import Float
from torch import Tensor

class ProjectionPreHookLiReFs:
  def __init__(
    self,
    direction: Float[Tensor, "n_embd"], 
    scale: float = 0.1
  ):
    super().__init__()
    self.direction = direction
    self.scale = scale

  def __call__(
    self,
    module,
    input,
  ):
    projection = (input[0] @ self.direction).unsqueeze(-1) * self.direction
    input[0][:, :, :] = input[0] + self.scale * projection