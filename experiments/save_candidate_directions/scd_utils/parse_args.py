import argparse
from typing import TypedDict

class Args(TypedDict):

  device: str

  models_name: list[str]
  hidden_states_file_paths: list[str]

  output_path: str

def parse_args() -> Args:
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--device',
    type=str,
    help="Device to run the model on (e.g., 'cuda:0' or 'cpu')",
  )

  parser.add_argument(
    '--models_name',
    type=str,
    nargs='+',
    help="Names of the models to process",
  )
  parser.add_argument(
    '--hidden_states_file_paths',
    type=str,
    nargs='+',
    help="Paths to the hidden states files for each model",
  )

  parser.add_argument(
    '--output_path',
    type=str,
    help="Path to save the candidate directions",
  )
  
  args = parser.parse_args()

  return args.__dict__