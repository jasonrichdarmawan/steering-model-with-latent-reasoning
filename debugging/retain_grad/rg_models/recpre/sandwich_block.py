from .raven_config import RavenConfig

from torch import nn
from torch import Tensor
from torch.nn import RMSNorm
from jaxtyping import Float

class SandwichBlock(nn.Module):
  def __init__(
    self,
    config: RavenConfig,
    layer_id: int,
  ):
    super().__init__()
    self.layer_id = layer_id
    self.norm_1 = RMSNorm(
      normalized_shape=config.n_embd,
      eps=config.norm_eps
    )

  def forward(
    self,
    hidden_states: Float[Tensor, "batch seq_len n_embd"],
  ):
    hidden_states = self.norm_1(hidden_states)

    outputs = (hidden_states,)

    # must return a tuple, not a dict to be compatible with torch.nn.Module.register_full_backward_pre_hook
    return outputs