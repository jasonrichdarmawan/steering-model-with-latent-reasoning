from utils import get_n_layers

from jaxtyping import Float
from torch import Tensor
from enum import Enum

class ProcessHiddenStatesMode(Enum):
  """
  Note;
  - LiReFs uses the LAST_RESPONSE_TOKEN mode. The source code indicates
  that LiReFs only generate 1 response token (use model.forward(...) 
  function) for each query. The paper stated the following:
  "We focus on the residual stream of the last token of the user turn,
  as the point when the model is going to generate the first answer token"
  - Venhoff et al. (2025) uses the ALL_TOKENS mode.
  - Since ALL_TOKENS mode will result in different seq_len for each input,
    we will use running mean to compute the average hidden states
  
  Reference:
  1. LiReFs https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/LiReFs_storing_hs.ipynb
  2. Venhoff et al. (2025) 
     a. old repository https://github.com/jasonrichdarmawan/steering-thinking-models/blob/0d5091a66b509504b6c61ccb4acf2650c9ac2408/train-steering-vectors/train_vectors.py#L107-L140
     b. new repository https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/utils/utils.py#L366-L400
  """

  FIRST_ANSWER_TOKEN = "FIRST_ANSWER_TOKEN"
  ALL_TOKENS = "ALL_TOKENS"

  def __str__(self):
    return self.value

def process_hidden_states(
  model,
  mode: ProcessHiddenStatesMode,
  hidden_states: dict[int, Float[Tensor, "batch seq_len n_embd"]],
  attention_mask: Float[Tensor, "batch seq_len"] | None = None,
  processed_hidden_states: dict[int, list[Float[Tensor, "seq_len n_embd"]]] | None = None,
):
  n_layers_to_cache = range(get_n_layers(model))

  if processed_hidden_states is None:
    processed_hidden_states = {
      layer_index: []
      for layer_index in n_layers_to_cache
    }

  n_batches_to_cache = range(hidden_states[0].shape[0])
  match mode:
    case ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN:
      for layer_index in n_layers_to_cache:
        for batch_index in n_batches_to_cache:
          hidden_state = hidden_states[layer_index][batch_index, -1].clone()
          processed_hidden_states[layer_index].append(hidden_state)
    case ProcessHiddenStatesMode.ALL_TOKENS:
      for layer_index in n_layers_to_cache:
        for batch_index in n_batches_to_cache:
          pad_len = (attention_mask[batch_index] == 0).sum()
          hidden_state = hidden_states[layer_index][batch_index, pad_len:].clone()
          processed_hidden_states[layer_index].append(hidden_state)

  return processed_hidden_states