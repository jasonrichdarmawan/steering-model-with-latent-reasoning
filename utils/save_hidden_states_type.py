from enum import Enum
from typing import TypedDict
from jaxtyping import Float
import torch as t

class SaveHiddenStatesQueryLabel(Enum):
  REASONING = "REASONING"
  MEMORIZING = "MEMORIZING"

  def __str__(self):
    return self.value

class SaveHiddenStatesOutput(TypedDict):
  queries: list[str]
  query_token_lengths: list[int]
  labels: list[SaveHiddenStatesQueryLabel]
  hidden_states: dict[int, list[Float[t.Tensor, "seq_len n_embd"]]]