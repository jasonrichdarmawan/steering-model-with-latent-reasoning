from transformers import PretrainedConfig

class RavenConfig(PretrainedConfig):
  model_type = "raven_muginn"
  keys_to_ignore_at_inference = []
  attribute_map = {
    "num_attention_heads": "n_heads",
    "hidden_size": "n_embd",
    "num_hidden_layers": "n_layers",
  }

  def __init__(
    self,
    tie_word_embeddings: bool,
    n_layers_in_prelude: int,
    n_layers_in_recurrent_block: int,
    mean_recurrence: int,
    n_layers_in_coda: int,
    n_layers: 8,
    effective_expected_depth: int,
    vocab_size: int,
    n_embd: int,
    norm_eps: float,
    **kwargs,
  ):
    super().__init__(
      tie_word_embeddings=tie_word_embeddings,
      **kwargs,
    )
    self.n_layers_in_prelude = n_layers_in_prelude
    self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
    self.n_layers = n_layers
    self.n_layers_in_coda = n_layers_in_coda
    self.effective_expected_depth = effective_expected_depth
    self.mean_recurrence = mean_recurrence
    self.vocab_size = self.padded_vocab_size = vocab_size
    self.n_embd = n_embd
    self.norm_eps = norm_eps