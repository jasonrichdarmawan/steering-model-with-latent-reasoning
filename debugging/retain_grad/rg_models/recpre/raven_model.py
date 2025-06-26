from .raven_pre_trained_model import RavenPreTrainedModel
from .sandwich_block import SandwichBlock

from torch import nn
from torch import Tensor
from torch.nn import RMSNorm
from jaxtyping import Float
from transformers.modeling_outputs import BaseModelOutputWithPast

class RavenModel(RavenPreTrainedModel):
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    self.wte = nn.Embedding(
      num_embeddings=config.padded_vocab_size,
      embedding_dim=config.n_embd,
    )

    self.prelude = nn.ModuleList(
      SandwichBlock(config=config, layer_id=i)
      for i in range(config.n_layers_in_prelude)
    )
    
    o = config.n_layers_in_prelude + config.n_layers_in_recurrent_block * config.mean_recurrence
    
    self.coda = nn.ModuleList(
      SandwichBlock(config=config, layer_id=i + o)
      for i in range(config.n_layers_in_coda)
    )

    self.ln_f = RMSNorm(
      normalized_shape=config.n_embd,
      eps=config.norm_eps
    )

    self.post_init()

  def get_input_embeddings(self):
    return self.wte
  
  def set_input_embeddings(self, value):
    self.wte = value
  
  def forward(self, input_ids: Float[Tensor, "batch seq_len"]):
    all_hidden_states = {}

    input_embeds: Float[Tensor, "batch seq_len n_embd"] = self.wte(input_ids)
    
    hidden_state = input_embeds

    for block in self.prelude:
      layer_outputs = block(
        hidden_states=hidden_state,
      )
      hidden_state = layer_outputs[0]

      all_hidden_states[block.layer_id] = hidden_state

    for block in self.coda:
      layer_outputs = block(
        hidden_states=hidden_state,
      )
      hidden_state = layer_outputs[0]

      all_hidden_states[block.layer_id] = hidden_state

    hidden_state = self.ln_f(hidden_state)
    if hidden_state.requires_grad:
      hidden_state.retain_grad()
    all_hidden_states[self.config.effective_expected_depth] = hidden_state

    return BaseModelOutputWithPast(
      last_hidden_state=hidden_state,
      hidden_states=all_hidden_states,
    )