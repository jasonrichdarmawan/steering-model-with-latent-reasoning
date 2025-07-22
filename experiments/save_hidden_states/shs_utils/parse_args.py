from utils import CacheHiddenStatesMode

import argparse
from enum import Enum
from typing import TypedDict

class Args(TypedDict):
  models_path: str
  model_name: str

  data_path: str
  data_name: str
  data_sample_size: int | None
  data_batch_size: int

  cache_hidden_states_mode: CacheHiddenStatesMode

  output_path: str

def parse_args() -> Args:
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--models_path',
    type=str,
    help="Path to the root directory containing multiple model folders",
  )
  parser.add_argument(
    '--model_name',
    type=str,
    help="Folder name of the specific model to load from the root directory",
  )

  parser.add_argument(
    '--data_path',
    type=str,
    help="Path to the root directory containing multiple data folders.",
  )
  parser.add_argument(
    '--data_name',
    type=str,
    help="Folder name of the specific dataset to load from the root directory.",
  )
  parser.add_argument(
    '--data_sample_size',
    type=int,
    help="Sample size of the dataset to load. If not specified, the entire dataset is loaded.",
    default=None,
  )
  parser.add_argument(
    '--data_batch_size',
    type=int,
    help="Batch size for processing the dataset.",
  )

  parser.add_argument(
    '--cache_hidden_states_mode',
    type=CacheHiddenStatesMode,
    choices=list(CacheHiddenStatesMode),
    help="Mode for caching hidden states.",
  )

  parser.add_argument(
    '--output_path',
    type=str,
    help="Path to save the output files.",
  )

  args = parser.parse_args()

  return args.__dict__