from utils import get_n_layers

from transformers import PreTrainedTokenizerBase
import torch
from jaxtyping import Float
from torch import Tensor

def cache_hidden_states(
  model,
  tokenizer: PreTrainedTokenizerBase,
  queries_batch: list[list[str]],
  hidden_states_cache: dict[int, Float[Tensor, "batch n_embd"]] | None = None,
):
  match model.config.model_type:
    case "huginn_raven":
      hidden_states_cache = _cache_hidden_states_huginn(
        model=model,
        tokenizer=tokenizer,
        queries_batch=queries_batch,
        hidden_states_cache=hidden_states_cache,
      )
    case "llama":
      hidden_states_cache = _cache_hidden_states_llama(
        model=model,
        tokenizer=tokenizer,
        queries_batch=queries_batch,
        hidden_states_cache=hidden_states_cache,
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")

  return hidden_states_cache

def get_n_embd(model) -> int:
  match model.config.model_type:
    case "huginn_raven":
      return model.config.n_embd
    case "llama":
      return model.config.hidden_size
    case _:
      raise ValueError(f"Model type {model.config.model_type} is not supported for n_embd retrieval.")
    
def _cache_hidden_states_llama(
  model,
  tokenizer: PreTrainedTokenizerBase,
  queries_batch: list[list[str]],
  hidden_states_cache: dict[int, Float[Tensor, "batch n_embd"]] | None = None,
):
  inputs = tokenizer(
    queries_batch,
    return_tensors="pt",
    padding="longest",
    return_token_type_ids=False,
  ).input_ids.to(device=model.device)

  with torch.no_grad():
    # Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/LiReFs_storing_hs.ipynb
    # TODO: Why use model.forward instead of model.generate and only once?
    outputs = model(
      inputs,
      output_hidden_states=True,
    )

  n_layers_to_cache = range(get_n_layers(model))

  if hidden_states_cache is None:
    n_embd = get_n_embd(model)

    hidden_states_cache = {
      index: torch.empty(
        (0, n_embd),
        dtype=model.dtype,
        device="cpu",
      )
      for index in n_layers_to_cache
    }
  
  for layer_index in n_layers_to_cache:
    hidden_state = outputs.hidden_states[layer_index]

    hidden_states_cache[layer_index] = torch.cat(
      (
        hidden_states_cache[layer_index],
        hidden_state[:, -1, :].detach().clone().cpu()
      ),
      dim=0
    )

  return hidden_states_cache
    
def _cache_hidden_states_huginn(
  model,
  tokenizer: PreTrainedTokenizerBase,
  queries_batch: list[list[str]],
  hidden_states_cache: dict[int, Float[Tensor, "batch n_embd"]] | None = None,
):
  inputs = tokenizer(
    queries_batch,
    return_tensors="pt",
    padding="longest",
    return_token_type_ids=False
  ).input_ids.to(device=model.device)

  with torch.no_grad():
    # TODO: use model.generate too,
    # using model.forward with question as inputs to compute steering vectors
    # does not make sense. Although LiReFs claims that this method improves the
    # model final answer accuracy by 7.2%, I am going to reproduce it first
    # before proceeding with this plan.
    #
    # 1st step should be create to a new job, to save the model's responses.
    # This API must not be changed for backward compatibility.
    # Heck, no ones using this code yet. But, I was a Software Engineer,
    # my ego is telling me to make it backward compatible.
    # This is against the empirical alignment research principle,
    # "If we talk about an idea, you can tell me whether it works the next day".
    # This is my first research, so I am not going to follow this principle.
    # For my future research, I will follow this principle.
    #
    # Note:
    # 1. LiReFs uses model.forward
    # 2. Venhoff et al. (2025) uses model.generate and then uses model.forward to get the hidden states
    #
    # Reference:
    # 1. LiReFs https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/LiReFs_storing_hs.ipynb
    # 2. Venhoff et al. (2025) 
    #    a. old repository https://github.com/jasonrichdarmawan/steering-thinking-models/blob/0d5091a66b509504b6c61ccb4acf2650c9ac2408/train-steering-vectors/train_vectors.py#L107-L140
    #    b. new repository https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/utils/utils.py#L366-L400
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
  
  n_layers_to_cache = range(get_n_layers(model))
  
  if hidden_states_cache is None:
    n_embd = get_n_embd(model)

    hidden_states_cache = {
      index: torch.empty(
        (0, n_embd),
        dtype=model.dtype,
        device="cpu",
      )
      for index in n_layers_to_cache
    }

  for layer_index in n_layers_to_cache:
    hidden_state = outputs["hidden_states"].get(layer_index)
    if hidden_state is None:
      continue

    hidden_states_cache[layer_index] = torch.cat(
      (
        hidden_states_cache[layer_index],
        hidden_state[:, -1, :].detach().clone().cpu()
      ),
      dim=0
    )
  
  return hidden_states_cache