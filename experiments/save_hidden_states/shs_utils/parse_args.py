import argparse
from typing import TypedDict

class Args(TypedDict):
  models_path: str
  model_name: str

  data_path: str
  data_name: str

  data_sample_size: int | None
  data_batch_size: int

  output_file_path: str

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
    default="huginn-0125",
  )
  
  parser.add_argument(
    '--data_path',
    type=str,
    help="Path to the root directory containing multiple data folders.",
  )
  parser.add_argument(
    '--data_name',
    type=str,
    help="Folder name of the specific dataset to load from the root directory, or the key for hidden_states_cache.",
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
    default=4,
  )
  
  parser.add_argument(
    '--output_file_path',
    type=str,
    help="Path to save the cached hidden states",
  )
  
  args, _ = parser.parse_known_args()

  return args.__dict__