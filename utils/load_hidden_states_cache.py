from typing import Any

import torch

def load_hidden_states_cache(file_path: str) -> dict[str, Any]:
  """
  Load the hidden states cache from a file.

  Args:
    file_path (str): Path to the hidden states cache file.

  Returns:
    dict: The loaded hidden states cache.
  """
  hidden_states_cache = torch.load(file_path, map_location="cpu")
  return hidden_states_cache