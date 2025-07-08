from jobs import get_save_candidate_directions
from jobs import get_analyze_steering_effect_per_layer
from jobs import get_evaluate_lm_eval
from jobs import get_evaluate_accuracy_reasoning_memorizing

def get_mmlu_pro(
  workspace_path: str,
  job: str,
) -> list[str]:
  
  jobs = []
  
  if job == "mmlu_pro" or job == "mmlu_pro_save_candidate_directions":
    job = get_save_candidate_directions(
      workspace_path=workspace_path,
      model_name="huginn-0125",
    )
    jobs.append(job)
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_save_candidate_directions":
    job = get_save_candidate_directions(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
    )
    jobs.append(job)

  if job == "mmlu_pro" or job == "mmlu_pro_analyze_steering_effect_per_layer":
    job = get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
    )
    jobs.append(job)

  # Huginn-0125 specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing":
    job = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=False,
    )
    jobs.append(job)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention":
    job = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,
      layer_indices=[66],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    jobs.append(job)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_129":
    job = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,
      layer_indices=[129],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    jobs.append(job)
  
  # Meta-Llama-3-8B specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_evaluate_accuracy_reasoning_memorizing":
    job = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=False,
    )
    jobs.append(job)
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_evaluate_accuracy_reasoning_memorizing_with_intervention":
    job = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,
      layer_indices=[21],
      with_hidden_states_pre_hook=True,
      with_hidden_states_post_hook=False,
    )
    jobs.append(job)

  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval":
    job = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu_pro",
      with_intervention=False,
    )
    jobs.append(job)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval_with_intervention":
    job = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu_pro",
      with_intervention=True,
      layer_indices=[66],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    jobs.append(job)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval_with_intervention_129":
    job = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu_pro",
      with_intervention=True,
      layer_indices=[129],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    jobs.append(job)

  if len(jobs) == 0:
    raise ValueError(f"Job '{job}' not found in MMLU Pro jobs.")

  return jobs