from jobs import get_mmlu, get_mmlu_pro

def set_up_experiments(
  workspace_path: str,
  jobs: list[str]
) -> list[list[str]]:
  experiments = []

  for job in jobs:
    if job.startswith("mmlu_pro"):
      commands = get_mmlu_pro(workspace_path=workspace_path, job=job,)
      experiments.append(commands)
    elif job.startswith("mmlu"):
      commands = get_mmlu(workspace_path=workspace_path, job=job,)
      experiments.append(commands)
    else:
      raise ValueError(f"Unknown job: {job}")

  return experiments