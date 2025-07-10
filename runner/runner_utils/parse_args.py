from typing import TypedDict
import argparse

class Args(TypedDict):
  workspace_path: str
  output_path: str | None
  shutdown_after_experiment: bool

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Run multiple experiments in sequence and log the results."
  )

  parser.add_argument(
    '--workspace_path',
    type=str,
    help="Path to the workspace directory where experiments will be run.",
  )
  
  parser.add_argument(
    '--jobs',
    type=str,
    nargs='+',
    help="List of jobs to run. Each job should be a string representing the job name.",
  )
  
  parser.add_argument(
    '--output_path',
    type=str,
    help="Path to save the output log file.",
    default=None,
  )

  parser.add_argument(
    '--shutdown_after_experiment',
    action='store_true',
    help="If set, the script will shut down after running the experiments.",
  )
  
  args = parser.parse_args()
  
  return args.__dict__