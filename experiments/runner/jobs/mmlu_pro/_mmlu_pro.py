from .mmlu_pro_save_hidden_states import get_mmlu_pro_save_hidden_states
from .mmlu_pro_evaluate_accuracy_reasoning_memorizing import get_mmlu_pro_evaluate_accuracy_reasoning_memorizing
from .mmlu_pro_evaluate_lm_eval import get_mmlu_pro_evaluate_lm_eval
from .mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention import get_mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention

def get_mmlu_pro(
  workspace_path: str,
  job: str,
) -> list[str]:
  
  jobs = []
  
  if "mmlu_pro_save_hidden_states".startswith(job):
    jobs.append(get_mmlu_pro_save_hidden_states(workspace_path=workspace_path))
  if "mmlu_pro_evaluate_accuracy_reasoning_memorizing".startswith(job):
    jobs.append(get_mmlu_pro_evaluate_accuracy_reasoning_memorizing(workspace_path=workspace_path))
  if "mmlu_pro_evaluate_lm_eval".startswith(job):
    jobs.append(get_mmlu_pro_evaluate_lm_eval(workspace_path=workspace_path))
  if "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention".startswith(job):
    jobs.append(get_mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention(workspace_path=workspace_path))

  return jobs