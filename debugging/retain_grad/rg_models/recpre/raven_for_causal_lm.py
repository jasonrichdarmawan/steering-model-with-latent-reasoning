from .raven_pre_trained_model import RavenPreTrainedModel
from .raven_config import RavenConfig
from .raven_model import RavenModel

from torch import Tensor
from torch import nn
from jaxtyping import Float
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast

class RavenForCausalLM(RavenPreTrainedModel):
  _tied_weights_keys = ["lm_head.weight"]
  
  def __init__(
    self,
    config: RavenConfig,
  ):
    super().__init__(config)
    self.model = RavenModel(config)
    self.lm_head = nn.Linear(
      in_features=config.n_embd,
      out_features=config.padded_vocab_size,
      bias=False,
    )
    self.post_init()

  def get_input_embeddings(self):
    return self.model.wte
  
  def set_input_embeddings(self, value):
    self.model.wte = value

  def get_output_embeddings(self):
    return self.lm_head
  
  def set_output_embeddings(self, value):
    self.lm_head = value

  def get_decoder(self):
    return self.model

  def set_decoder(self, decoder):
    self.model = decoder

  def forward(self, input_ids: Float[Tensor, "batch seq_len"]):
    outputs: BaseModelOutputWithPast = self.model(input_ids)

    hidden_state = outputs.last_hidden_state
    logits = self.lm_head(hidden_state)

    return CausalLMOutputWithPast(
      logits=logits,
      hidden_states=outputs.hidden_states,
    )
    