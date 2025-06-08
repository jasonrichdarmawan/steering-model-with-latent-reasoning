import argparse
from typing import TypedDict

class Args(TypedDict):
  models_path: str
  model_name: str
  data_path: str
  data_name: str

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
  
  args, _ = parser.parse_known_args()

  return {
    "bebe": 12,
    "models_path": args.models_path,
    "model_name": args.model_name,
    "data_path": args.data_path,
    "data_name": args.data_name,
  }