from jobs import get_evaluate_lm_eval

def get_mmlu(
  workspace_path: str,
  job: str,
) -> list[str]:
  jobs = []

  if job == "mmlu" or job == "mmlu_evaluate_lm_eval":
    job = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=False,
    )
    jobs.append(job)
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention":
    job = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=True,
      layer_indices=[66],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    jobs.append(job)
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention_129":
    job = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=True,
      layer_indices=[129],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    jobs.append(job)
  
  if len(jobs) == 0:
    raise ValueError(f"Job '{job}' not found in MMLU jobs.")

  return jobs