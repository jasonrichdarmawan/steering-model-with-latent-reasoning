# %%

import os
import sys

# To be able to import modules from the utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")
  sys.path.insert(0, project_root)

# %%

if False:
  from utils import reload_modules
  
  reload_modules(
    project_root=project_root,
  )

from runner_utils import parse_args
from runner_utils import set_up_experiments

from typing import TypedDict
from datetime import datetime
from typing import Optional
import os
import subprocess
import sys

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  root_path = "/media/npu-tao/disk4T/jason"
  sys.argv = [
    'main.py',
    '--workspace_path', root_path,
    '--jobs', 'mmlu_pro_save_candidate_directions', 'mmlu_pro_meta-llama-3-8b_save_candidate_directions',
    # '--output_path', f'{root_path}/experiments/runner',
    # "--shutdown_after_experiment",
  ]

args = parse_args()
print(f"Parsed arguments:")
print('#' * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

experiments = set_up_experiments(
  workspace_path=args['workspace_path'],
  jobs=args["jobs"],
)
print(f"Set up jobs for experiments:")
print('#' * 60)
for i, commands in enumerate(experiments):
  print(f"Experiment {i+1}:")
  for j, command in enumerate(commands):
    print(f"Command {j+1}:\n{command}")

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

if args['output_path']:
  now = datetime.now()
  timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
  file_path = f"{args['output_path']}/{timestamp}.txt"

  os.makedirs(args['output_path'], exist_ok=True)  # Ensure the output directory exists

  f = open(file_path, 'w', encoding='utf-8')
  original_stdout = sys.stdout
  original_stderr = sys.stderr
  tee = Tee(original_stdout, f)
  sys.stdout = tee # Redirect stdout to both console and file
  sys.stderr = tee

  print(f"Output will be saved to: {file_path}")
else:
  print("No output file path specified, results will only be printed to the console.")

# %%

for i, commands in enumerate(experiments):
  print(f"\n\nExperiment {i+1}")
  print("#" * 60)
  failed = False
  failed_at = None
  for j, command in enumerate(commands):
    print(f"\nRunning command {j+1}/{len(commands)}.")
    print("#" * 60)
    print(f"Command:\n{command}")
    process = subprocess.Popen(
      command,
      cwd=project_root,
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
      print(f"{'#' * 30} Command {j+1}/{len(commands)} failed with return code {returncode}. Skipping to the next experiment. {'#' * 30}")
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

# %%

print("\n\nExperiment Results Summary")
print("#" * 60)
for result in experiment_results:
  if result["status"] == 1:
    print(f"Experiment {result['experiment_number']}: Success")
  else:
    print(f"Experiment {result['experiment_number']}: Failed at command {result['failed_at']}")

# %%

if args['output_path']:
  print(f"\n\nAll results have been saved to: {file_path}")
  sys.stdout = original_stdout
  sys.stderr = original_stderr
  f.close()

# %%

if args["shutdown_after_experiment"]:
  print("Shutting down the system after experiments.")
  subprocess.run(["/usr/bin/shutdown"], check=True)

# %%