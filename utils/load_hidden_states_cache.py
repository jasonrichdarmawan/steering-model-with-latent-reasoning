from typing import Any

import torch

import os

def load_hidden_states_cache(hidden_states_cache_path: str, model_name: str) -> dict[str, Any]:
  """
  Load the hidden states cache from a file.

  Args:
      hidden_states_cache_file_path (str): Path to the hidden states cache file.

  Returns:
      dict: The loaded hidden states cache.
  """
  hidden_states_cache_file_path = os.path.join(hidden_states_cache_path, f"{model_name}_hidden_states_cache.pt")
  try:
    hidden_states_cache = torch.load(hidden_states_cache_file_path, map_location='cpu')
    return hidden_states_cache
  except FileNotFoundError:
    print(f"Error: The file {hidden_states_cache_file_path} does not exist.")
    return {}