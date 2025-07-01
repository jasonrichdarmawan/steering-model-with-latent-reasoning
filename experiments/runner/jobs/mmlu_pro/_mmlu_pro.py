from .mmlu_pro_save_hidden_states import get_mmlu_pro_save_hidden_states
from .mmlu_pro_analyze_steering_effect_per_layer import get_mmlu_pro_analyze_steering_effect_per_layer
from .mmlu_pro_evaluate_accuracy_reasoning_memorizing import get_mmlu_pro_evaluate_accuracy_reasoning_memorizing
from .mmlu_pro_evaluate_lm_eval import get_mmlu_pro_evaluate_lm_eval
from .mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention import get_mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention
from .mmlu_pro_evaluate_lm_eval_with_intervention import get_mmlu_pro_evaluate_lm_eval_with_intervention

def get_mmlu_pro(
  workspace_path: str,
  job: str,
) -> list[str]:
  
  jobs = []
  
  if job == "mmlu_pro" or job == "mmlu_pro_save_hidden_states":
    jobs.append(get_mmlu_pro_save_hidden_states(workspace_path=workspace_path))
  if job == "mmlu_pro" or job == "mmlu_pro_analyze_steering_effect_per_layer":
    jobs.append(get_mmlu_pro_analyze_steering_effect_per_layer(workspace_path=workspace_path))
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing":
    jobs.append(get_mmlu_pro_evaluate_accuracy_reasoning_memorizing(workspace_path=workspace_path))
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention":
    jobs.append(
      get_mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention(
        workspace_path=workspace_path,
        layer_indices=[66],
      )
    )
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_129":
    jobs.append(
      get_mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention(
        workspace_path=workspace_path,
        layer_indices=[129],
      )
    )
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval":
    jobs.append(get_mmlu_pro_evaluate_lm_eval(workspace_path=workspace_path))
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval_with_intervention":
    jobs.append(get_mmlu_pro_evaluate_lm_eval_with_intervention(workspace_path=workspace_path))

  if len(jobs) == 0:
    raise ValueError(f"Job '{job}' not found in MMLU Pro jobs.")

  return jobs