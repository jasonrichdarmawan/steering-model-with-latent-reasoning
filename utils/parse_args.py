import argparse
from typing import TypedDict

class Args(TypedDict):
  models_path: str
  model_name: str

def parse_args() -> Args:
  parser = argparse.ArgumentParser()

  parser.add_argument('--models_path', type=str,
                      help="Path to the root directory containing multiple model folders")
  parser.add_argument('--model_name', type=str,
                      help="Folder name of the specific model to load from the root directory")
  
  args = parser.parse_args()
  return {
    "models_path": args.models_path,
    "model_name": args.model_name,
  }