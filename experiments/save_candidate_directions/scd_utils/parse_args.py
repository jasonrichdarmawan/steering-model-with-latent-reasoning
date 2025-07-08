from utils import ProcessHiddenStatesMode

import argparse
from typing import TypedDict

class Args(TypedDict):
  models_path: str
  model_name: str

  device_map: str | None

  huginn_num_steps: int | None

  data_path: str
  data_name: str

  data_sample_size: int | None
  data_batch_size: int

  process_hidden_states_mode: ProcessHiddenStatesMode

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
    '--device_map',
    type=str,
    help="Device map for the model. If not specified, the model will be loaded on the CPU.",
    default="cuda",
  )

  parser.add_argument(
    '--huginn_num_steps',
    type=int,
    help="Number of steps for Huginn model generation. If the model is not Huginn, this argument is ignored.",
    default=None,
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
    help="Number of samples to randomly select from the test dataset",
    default=None,
  )
  parser.add_argument(
    '--data_batch_size',
    type=int,
    help="Batch size for processing the data. Increase this value to maximize GPU utilization (GPU-Util). If GPU-Util reaches 100% consistently, you have found the optimal batch size; increasing it further may cause torch.OutOfMemoryError. This does not affect the values of hidden states.",
  )

  parser.add_argument(
    '--process_hidden_states_mode',
    type=ProcessHiddenStatesMode,
    choices=list(ProcessHiddenStatesMode),
    help="Mode for processing hidden states. 'FIRST_ANSWER_TOKEN' process the hidden states of the first answer token, while 'ALL_TOKENS' process the hidden states of all tokens.",
  )
  
  parser.add_argument(
    '--output_path',
    type=str,
    help="Path to save the candidate directions",
  )
  
  args = parser.parse_args()

  return args.__dict__