from .mmlu_evaluate_lm_eval import get_mmlu_evaluate_lm_eval
from .mmlu_evaluate_lm_eval_with_intervention import get_mmlu_evaluate_lm_eval_with_intervention

def get_mmlu(
  workspace_path: str,
  job: str,
) -> list[str]:
  jobs = []

  if job == "mmlu" or job == "mmlu_evaluate_lm_eval":
    jobs.append(get_mmlu_evaluate_lm_eval(workspace_path=workspace_path))
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention":
    jobs.append(get_mmlu_evaluate_lm_eval_with_intervention(workspace_path=workspace_path))
  
  if len(jobs) == 0:
    raise ValueError(f"Job '{job}' not found in MMLU jobs.")

  return jobs