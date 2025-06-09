import argparse
from typing import TypedDict

class Args(TypedDict):
  models_path: str
  model_name: str
  data_path: str
  data_name: str
  data_sample_size: int
  data_batch_size: int
  output_path: str

def parse_args() -> Args:
  parser = argparse.ArgumentParser()

  parser.add_argument('--models_path', type=str,
                      help="Path to the root directory containing multiple model folders",
                      default="/root/autodl-fs/transformers")
  parser.add_argument('--model_name', type=str,
                      help="Folder name of the specific model to load from the root directory",
                      default="huginn-0125")
  
  parser.add_argument('--data_path', type=str,
                      help="Path to the root directory containing multiple data folders",
                      default="/root/autodl-fs/datasets")
  parser.add_argument('--data_name', type=str,
                      help="Folder name of the specific dataset to load from the root directory",
                      default="mmlu-pro")
  parser.add_argument('--data_sample_size', type=int,
                      help="Number of samples to randomly select from the test dataset",
                      default=600)
  parser.add_argument('--data_batch_size', type=int,
                      help="Batch size for processing the data",
                      default=4)
  
  parser.add_argument('--output_path', type=str,
                      help="Path to save the cached hidden states",
                      default="/root/autodl-fs/hidden_states_cache")
  
  args, _ = parser.parse_known_args()

  return {
    "models_path": args.models_path,
    "model_name": args.model_name,
    "data_path": args.data_path,
    "data_name": args.data_name,
    "data_sample_size": args.data_sample_size,
    "data_batch_size": args.data_batch_size,
    "output_path": args.output_path,
  }