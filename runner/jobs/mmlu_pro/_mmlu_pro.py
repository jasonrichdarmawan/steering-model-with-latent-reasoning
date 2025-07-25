from utils import ProcessHiddenStatesMode
from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode

from jobs import get_save_candidate_directions
from jobs import get_analyze_steering_effect_per_layer
from jobs import get_evaluate_lm_eval
from jobs import get_evaluate_accuracy_reasoning_memorizing

def get_mmlu_pro(
  workspace_path: str,
  job: str,
) -> list[str]:
  
  commands = []
  
  # huginn-0125 specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_save_candidate_directions":
    command = get_save_candidate_directions(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_save_candidate_directions_all_tokens":
    command = get_save_candidate_directions(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      process_hidden_states_mode=ProcessHiddenStatesMode.ALL_TOKENS,
    )
    commands.append(command)
  
  # Meta-LLama-3-8B specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_save_candidate_directions":
    command = get_save_candidate_directions(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_save_candidate_directions_all_tokens":
    command = get_save_candidate_directions(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      process_hidden_states_mode=ProcessHiddenStatesMode.ALL_TOKENS,
    )
    commands.append(command)

  # huginn-0125 specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_analyze_steering_effect_per_layer":
    command = get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_analyze_steering_effect_per_layer_all_tokens":
    command = get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
      process_hidden_states_mode=ProcessHiddenStatesMode.ALL_TOKENS,
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
    )
    commands.append(command)
  
  # Meta-Llama-3-8B specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_analyze_steering_effect_per_layer":
    command = get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_analyze_steering_effect_per_layer_all_tokens":
    command = get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
      process_hidden_states_mode=ProcessHiddenStatesMode.ALL_TOKENS,
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
    )
    commands.append(command)

  # Huginn-0125 specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=False,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      
      layer_indices=[66],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_129":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,
      
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      
      layer_indices=[129],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_1":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,
      
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      
      layer_indices=[1],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_1_all_tokens":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,

      process_hidden_states_mode=ProcessHiddenStatesMode.ALL_TOKENS,
      
      layer_indices=[1],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
    commands.append(command)
  
  # Meta-Llama-3-8B specific jobs
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_evaluate_accuracy_reasoning_memorizing":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=False,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_meta-llama-3-8b_evaluate_accuracy_reasoning_memorizing_with_intervention":
    command = get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro",
      with_intervention=True,

      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      
      layer_indices=[21],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=True,
      with_hidden_states_post_hook=False,
      scale=1.0,
    )
    commands.append(command)

  # Evaluate lm_eval jobs
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu_pro",
      with_intervention=False,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval_with_intervention":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu_pro",
      with_intervention=True,
      
      use_candidate_directions=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      
      layer_indices=[66],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
    commands.append(command)
  if job == "mmlu_pro" or job == "mmlu_pro_evaluate_lm_eval_with_intervention_129":
    command = get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="mmlu_pro",
      with_intervention=True,
      
      use_candidate_directions=True,
      process_hidden_states_mode=ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN,
      
      layer_indices=[129],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
    commands.append(command)

  if len(commands) == 0:
    raise ValueError(f"Job '{job}' not found in MMLU Pro jobs.")

  return commands