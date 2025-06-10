from typing import Any

import torch

def load_hidden_states_cache(hidden_states_cache_file_path: str) -> dict[str, Any]:
  """
  Load the hidden states cache from a file.

  Args:
      hidden_states_cache_file_path (str): Path to the hidden states cache file.

  Returns:
      dict: The loaded hidden states cache.
  """
  try:
    hidden_states_cache = torch.load(hidden_states_cache_file_path, map_location='cpu')
    return hidden_states_cache
  except FileNotFoundError:
    print(f"Error: The file {hidden_states_cache_file_path} does not exist.")
    return {}