from .raven_pre_trained_model import *
from .raven_config import *
from .raven_model import *
from .raven_for_causal_lm import *

from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM

AutoConfig.register(
  model_type="raven_muginn",
  config=RavenConfig,
)
AutoModel.register(
  config_class=RavenConfig,
  model_class=RavenForCausalLM,
)
AutoModelForCausalLM.register(
  config_class=RavenConfig,
  model_class=RavenForCausalLM,
)