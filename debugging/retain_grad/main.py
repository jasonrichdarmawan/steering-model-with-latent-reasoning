# %%

import os
import sys

# To be able to import modules from the utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")
  sys.path.insert(0, project_root)

# %%

if True:
  print("Reloading modules to ensure the latest code is used.")
  import sys
  from importlib import reload
  reload(sys.modules.get("rg_models.recpre.raven_config", sys))
  reload(sys.modules.get("rg_models.recpre.sandwich_block", sys))
  reload(sys.modules.get("rg_models.recpre.raven_model", sys))
  reload(sys.modules.get("rg_models.recpre.raven_for_causal_lm", sys))
  reload(sys.modules.get("rg_models.recpre", sys))
  reload(sys.modules.get("rg_models", sys))

from rg_models import RavenConfig

from transformers import AutoModelForCausalLM
from accelerate import dispatch_model
import torch
from torch import nn
from torch import Tensor

# %%

print("Initializing model...")
config = RavenConfig(
  tie_word_embeddings=True,
  n_layers_in_prelude=2,
  n_layers_in_recurrent_block=4,
  mean_recurrence=32,
  n_layers_in_coda=2,
  n_layers=8,
  effective_expected_depth=132,
  vocab_size=4,
  n_embd=16,
  norm_eps=1e-6,
)

model = AutoModelForCausalLM.from_config(
  config=config,
)
device_map = {
  "model.wte": 0,
  "model.prelude.0": 0,
  "model.prelude.1": 0,
  "model.coda": 1,
  "model.ln_f": 1,
  "lm_head": 0,
}
model = dispatch_model(
  model=model,
  device_map=device_map,
)

# %%

all_hidden_states = {}
def save_grad(module_name: str, layer_id: int):
  def hook(
    module: nn.Module,
    grad_output: Tensor,
  ):
    all_hidden_states[layer_id] = grad_output
  return hook

print("Registering hooks to save hidden states...")
hooks = []
for block in model.model.prelude:
  hook = block.register_full_backward_pre_hook(
    hook=save_grad(
      module_name="model.transformer.prelude",
      layer_id=block.layer_id
    ),
  )
  hooks.append(hook)
for block in model.model.coda:
  hook = block.register_full_backward_pre_hook(
    hook=save_grad(
      module_name="model.transformer.coda",
      layer_id=block.layer_id,
    ),
  )
  hooks.append(hook)

hook = model.model.ln_f.register_full_backward_pre_hook(
  hook=save_grad(
    module_name="transformer.ln_f", 
    layer_id=config.effective_expected_depth,
  ),
)
hooks.append(hook)

input_ids = torch.randint(
  low=0,
  high=config.vocab_size,
  size=(1, 4),
  device="cuda:0",
)
outputs = model(input_ids)

logits = outputs["logits"]

logits.sum().backward()

print("Removing hooks...")
for hook in hooks:
  hook.remove()

for layer_index, hidden_state in all_hidden_states.items():
  print(f"Layer {layer_index} hidden state grad:", hidden_state)

print("Logits:", logits)

# %%
