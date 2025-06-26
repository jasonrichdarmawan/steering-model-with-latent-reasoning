
from typing import TypedDict
import argparse

class Args(TypedDict):
  models_path: str
  model_name: str

  hidden_states_cache_file_path: str
  data_path: str
  data_name: str
  data_sample_size: int | None
  data_batch_size: int

  huginn_num_steps: int | None

  with_pre_hook: bool
  with_post_hook: bool
  projection_scale: float

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Parse arguments for ASEPL experiment."
  )
  
  parser.add_argument(
    '--models_path',
    type=str,
    required=True,
    help='Path to the directory containing the model files.'
  )
  parser.add_argument(
    '--model_name',
    type=str,
    required=True,
    help='Name of the model to be used in the experiment.'
  )

  parser.add_argument(
    '--hidden_states_cache_file_path',
    type=str,
    required=True,
    help='Path to the file where hidden states will be cached.'
  )
  parser.add_argument(
    '--data_path',
    type=str,
    required=True,
    help='Path to the directory containing the dataset files.'
  )
  parser.add_argument(
    '--data_name',
    type=str,
    required=True,
    help='Name of the dataset to be used in the experiment.'
  )
  parser.add_argument(
    '--data_sample_size',
    type=int,
    default=None,
    help='Number of samples to be used from the dataset. If None, all samples will be used.'
  )
  parser.add_argument(
    '--data_batch_size',
    type=int,
    default=4,
    help='Batch size for processing the dataset.'
  )

  parser.add_argument(
    '--huginn_num_steps',
    type=int,
    default=None,
    help='Number of steps for Huginn model generation. If None, the default value will be used.'
  )

  parser.add_argument(
    '--with_pre_hook',
    action='store_true',
    help='Whether to use a pre-hook for the intervention.'
  )
  parser.add_argument(
    '--with_post_hook',
    action='store_true',
    help='Whether to use a post-hook for the intervention.'
  )
  parser.add_argument(
    '--projection_scale',
    type=float,
    default=0.1,
    help='Scale factor for the projection in the intervention.'
  )

  args = parser.parse_args()
  
  return args.__dict__
