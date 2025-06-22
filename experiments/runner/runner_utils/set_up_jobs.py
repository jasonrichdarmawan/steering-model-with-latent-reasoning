from jobs import get_mmlu, get_mmlu_pro

def set_up_jobs(
  workspace_path: str,
  jobs: list[str]
) -> list[list[str]]:
  output = []

  for job in jobs:
    if job.startswith("mmlu_pro"):
      output.append(get_mmlu_pro(workspace_path=workspace_path, job=job,))
    elif job.startswith("mmlu"):
      output.append(get_mmlu(workspace_path=workspace_path, job=job,))
    else:
      raise ValueError(f"Unknown job: {job}")

  return output