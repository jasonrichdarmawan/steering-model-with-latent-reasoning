from transformers import PreTrainedTokenizerBase
import torch
from jaxtyping import Float
from torch import Tensor

def cache_hidden_states(
  model,
  tokenizer: PreTrainedTokenizerBase,
  queries_batch: list[list[str]],
  hidden_states_cache: dict[int, Float[Tensor, "seq_len n_embd"]] | None = None,
):
  if hidden_states_cache is None:
    n_layers = get_n_layers(model)
    n_embd = get_n_embd(model)

    hidden_states_cache = {
      index: torch.empty(
        (0, n_embd), dtype=model.dtype,
      )
      for index in range(n_layers)
    }

  match model.config.model_type:
    case name if name.startswith("huginn_"):
      hidden_states_cache = _cache_hidden_states_huginn(
        model=model,
        tokenizer=tokenizer,
        queries_batch=queries_batch,
        hidden_states_cache=hidden_states_cache,
      )

  return hidden_states_cache

def get_n_layers(model) -> int:
  match model.config.model_type:
    case name if name.startswith("huginn_"):
      return model.config.effective_expected_depth
    case _:
      print("Model type not recognized for n_layers retrieval.")
      return 0

def get_n_embd(model) -> int:
  match model.config.model_type:
    case name if name.startswith("huginn_"):
      return model.config.n_embd
    case _:
      print("Model type not recognized for n_embd retrieval.")
      return 0
    
def _cache_hidden_states_huginn(
  model,
  tokenizer: PreTrainedTokenizerBase,
  queries_batch: list[list[str]],
  hidden_states_cache: dict[int, Float[Tensor, "seq_len n_embd"]],
):
  n_layers_to_cache = list(
    range(get_n_layers(model))
  )

  inputs = tokenizer(
    queries_batch,
    return_tensors="pt",
    padding="longest",
    return_token_type_ids=False
  ).input_ids.to("cuda")

  with torch.no_grad():
    outputs = model(
      inputs,
      output_details={
        "return_logits": False,
        "return_latents": False,
        "return_attention": False,
        "return_head": True,
        "return_stats": False,
      }
    )

  for layer in n_layers_to_cache:
    # Tensor.detach() is important to avoid torch.OutOfMemoryError
    hidden_states_cache[layer] = torch.cat(
      (
        hidden_states_cache[layer],
        outputs["hidden_states"][layer][:, -1, :].detach().cpu()
      ),
      dim=0
    )
  
  return hidden_states_cache