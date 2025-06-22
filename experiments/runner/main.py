# %%

from runner_utils import parse_args
from runner_utils import set_up_jobs

from typing import TypedDict
from datetime import datetime
from typing import Optional
import os
import subprocess

if False:
  from importlib import reload
  import sys
  print("Reloading modules to ensure the latest code is used.")
  reload(sys.modules.get('runner_utils.parse_args', sys))
  reload(sys.modules.get('runner_utils.set_up_jobs', sys))
  reload(sys.modules.get('runner_utils', sys))

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/media/tao/disk4T/jason/recurrent-env"
  sys.argv = [
    'main.py',
    '--workspace_path', root_path,
    '--jobs', 'mmlu',
    '--output_path', f'{root_path}/experiments/runner',
  ]

args = parse_args()
print(f"Parsed arguments:")
print('#' * 60)
print(args)

# %%

EXPERIMENT_JOBS = set_up_jobs(
  workspace_path=args['workspace_path'],
  jobs=args["jobs"],
)
print(f"Set up jobs for experiments:")
print('#' * 60)
for i, job in enumerate(EXPERIMENT_JOBS):
  print(f"Experiment {i+1}:")
  for j, cmd in enumerate(job):
    print(f"Command {j+1}:\n{cmd}")

# %%

class Tee:
  def __init__(self, *files):
    self.files = files

  def write(self, obj):
    """Write to all files"""
    for f in self.files:
      f.write(obj)
      # Flush is important to ensure real-time output to the console
      f.flush()
  
  def flush(self):
    """Flush all files"""
    for f in self.files:
      f.flush()

class ExperimentResult(TypedDict):
  experiment_number: int
  status: int  # 1 for success, 0 for failure
  failed_at: Optional[int]  # None if successful, otherwise the command number that failed

experiment_results: list[ExperimentResult] = []

now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
file_path = f"{args['output_path']}/{timestamp}.txt"

os.makedirs(args['output_path'], exist_ok=True)  # Ensure the output directory exists

with open(file_path, 'w', encoding='utf-8') as f:
  original_stdout = sys.stdout
  original_stderr = sys.stderr
  tee = Tee(original_stdout, f)
  sys.stdout = tee # Redirect stdout to both console and file
  sys.stderr = tee

  for i, job in enumerate(EXPERIMENT_JOBS):
    print(f"\n\nExperiment {i+1}")
    print("#" * 60)
    failed = False
    failed_at = None
    for j, cmd in enumerate(job):
      print(f"\nRunning command {i+1}/{len(EXPERIMENT_JOBS)}.")
      print("#" * 60)
      print(f"Command: {cmd}")
      # Reference:
      # https://stackoverflow.com/a/21978778/13285583
      process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
      )

      for line in process.stdout:
        print(line, end='')

      process.wait()
      returncode = process.returncode
      if returncode != 0:
        print(f"{'#' * 30} Command {j+1}/{len(job)} failed with return code {returncode}. Skipping to the next experiment. {'#' * 30}")
        failed = True
        failed_at = j + 1
        break
    if not failed:
      print(f"{'#' * 30} Experiment {i+1} completed successfully. {'#' * 30}")
      experiment_results.append({
        "experiment_number": i + 1,
        "status": 1,
        "failed_at": None
      })
    else:
      print(f"{'#' * 30} Experiment {i+1} failed at command {failed_at}. {'#' * 30}")
      experiment_results.append({
        "experiment_number": i + 1,
        "status": 0,
        "failed_at": failed_at
      })

  print("\n\nExperiment Results Summary")
  print("#" * 60)
  for result in experiment_results:
    if result["status"] == 1:
      print(f"Experiment {result['experiment_number']}: Success")
    else:
      print(f"Experiment {result['experiment_number']}: Failed at command {result['failed_at']}")

  sys.stdout = original_stdout
  sys.stderr = original_stderr

# %%