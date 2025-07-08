import torch
from torch import Tensor
from jaxtyping import Float

def get_hidden_states(
  model,
  input_ids: Float[Tensor, "batch seq_len"],
  attention_mask: Float[Tensor, "batch seq_len"],
) -> dict[int, Float[Tensor, "batch seq_len n_embd"]]:
  match model.config.model_type:
    case "huginn_raven":
      hidden_states = _get_hidden_states_huginn(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
      return hidden_states
    case "llama":
      hidden_states = _get_hidden_states_llama(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
      return hidden_states
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")

def _get_hidden_states_huginn(
  model,
  input_ids: Float[Tensor, "batch seq_len"],
  attention_mask: Float[Tensor, "batch seq_len"],
):
  outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_details={
      "return_logits": False,
      "return_latents": False,
      "return_attention": False,
      "return_head": True,
      "return_stats": False,
    }
  )

  return outputs["hidden_states"]

def _get_hidden_states_llama(
  model,
  input_ids: Float[Tensor, "batch seq_len"],
  attention_mask: Float[Tensor, "batch seq_len"],
):
  outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
  )

  hidden_states = outputs["hidden_states"]
  hidden_states = {
    index: hidden_states[index]
    for index in range(len(hidden_states))
  }

  return hidden_states