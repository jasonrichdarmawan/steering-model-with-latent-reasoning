from .mmlu_evaluate_lm_eval import get_mmlu_evaluate_lm_eval

def get_mmlu(
  workspace_path: str,
  job: str,
) -> list[str]:
  jobs = []

  if "mmlu_evaluate_lm_eval".startswith(job):
    jobs.append(get_mmlu_evaluate_lm_eval(workspace_path=workspace_path))

  return jobs