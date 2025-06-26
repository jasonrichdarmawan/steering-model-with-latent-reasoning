from .raven_config import RavenConfig

from transformers import PreTrainedModel

class RavenPreTrainedModel(PreTrainedModel):
  config_class = RavenConfig