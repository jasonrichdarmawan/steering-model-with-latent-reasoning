from typing import TypedDict
import argparse

class Args(TypedDict):
  output_path: str

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Run multiple experiments in sequence and log the results."
  )
  
  parser.add_argument(
    '--output_path',
    type=str,
    help="Path to save the output log file.",
  )
  
  args = parser.parse_args()
  
  return args.__dict__