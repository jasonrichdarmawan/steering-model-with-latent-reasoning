from utils import ProcessHiddenStatesMode
from jobs import get_evaluate_lm_eval

def get_mmlu(
  workspace_path: str,
  job: str,
) -> list[str]:
  commands = []

  if job == "mmlu" or job == "mmlu_evaluate_lm_eval":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=False,
    )
    commands.append(command)
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      layer_indices=[66],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    commands.append(command)
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention_129":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      layer_indices=[129],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    commands.append(command)
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention_1":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      layer_indices=[1],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    commands.append(command)
  if job == "mmlu" or job == "mmlu_evaluate_lm_eval_with_intervention_1_all_tokens":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu",
      with_intervention=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.ALL_TOKENS,
      layer_indices=[1],
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
    )
    commands.append(command)
  
  if len(commands) == 0:
    raise ValueError(f"Job '{job}' not found in MMLU jobs.")

  return commands