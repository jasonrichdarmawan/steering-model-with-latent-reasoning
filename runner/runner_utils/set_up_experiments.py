from utils import CacheHiddenStatesMode

from jobs import get_save_hidden_states
from jobs import get_mmlu
from jobs import get_mmlu_pro

def set_up_experiments(
  workspace_path: str,
  jobs: list[str]
) -> list[list[str]]:
  experiments = []

  for job in jobs:
    if job == "save_hidden_states":
      commands = get_save_hidden_states(
        workspace_path=workspace_path,
        model_name="huginn-0125",
        cache_hidden_states_mode=CacheHiddenStatesMode.FIRST_ANSWER_TOKEN,
      )
      experiments.append(commands)
    elif job.startswith("mmlu_pro"):
      commands = get_mmlu_pro(workspace_path=workspace_path, job=job,)
      experiments.append(commands)
    elif job.startswith("mmlu"):
      commands = get_mmlu(workspace_path=workspace_path, job=job,)
      experiments.append(commands)
    else:
      raise ValueError(f"Unknown job: {job}")

  return experiments