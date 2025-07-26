from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/evaluate_lm_eval/main.py \
--data_path "$WORKSPACE_PATH/datasets" \
\
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
--device cuda:0 \
\
{huginn_mean_recurrence_arg} \
\
{with_intervention_flag} \
\
{use_linear_probes_flag} \
{linear_probes_file_path_arg} \
\
{use_candidate_directions_flag} \
{process_hidden_states_mode_arg} \
{candidate_directions_file_path_arg} \
\
{layer_indices_arg} \
{direction_normalization_mode_arg} \
{projection_hook_mode_arg} \
{modification_mode_arg} \
{with_hidden_states_pre_hook_flag} \
{with_hidden_states_post_hook_flag} \
{scale_arg} \
\
--tasks {tasks} \
{num_fewshot_arg} \
--batch_size auto \
\
--output_file_path "$WORKSPACE_PATH/experiments/lm_eval_results/{model_name}.json"
"""

def handle_evaluate_lm_eval(
  workspace_path: str,
  job: str,
) -> str:
  if job == "evaluate_lm_eval_model_name_Meta-Llama-3-8B":
    return get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      tasks="mmlu",
      with_intervention=False,
    )
  elif job == "evaluate_lm_eval_model_name_Meta-Llama-3-8B_with_intervention_user_linear_probes":
    return get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      tasks="mmlu",
      with_intervention=True,
      
      use_linear_probes=True,
      
      layer_indices=[12],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )
  elif job == "evaluate_lm_eval_tasks_piqa":
    return get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="piqa",
      num_fewshot=0,
      with_intervention=False,
    )
  elif job == "evaluate_lm_eval_tasks_piqa_use_linear_probes":
    return get_evaluate_lm_eval(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      tasks="piqa",
      num_fewshot=0,

      with_intervention=True,
      
      use_linear_probes=True,

      layer_indices=[31],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.LAST_TOKEN,
      with_hidden_states_pre_hook=False,
      with_hidden_states_post_hook=True,
      scale=1.0,
    )

def get_evaluate_lm_eval(
  workspace_path: str,
  model_name: str,
  tasks: str,
  with_intervention: bool,
  num_fewshot: int = 0,
  
  use_linear_probes: bool = False,

  use_candidate_directions: bool = False,

  direction_normalization_mode: DirectionNormalizationMode | None = None,
  layer_indices: list[int] | None = False,
  projection_hook_mode: ProjectionHookMode | None = None,
  modification_mode: TokenModificationMode | None = None,
  with_hidden_states_pre_hook: bool = False,
  with_hidden_states_post_hook: bool = False,
  scale: float | None = None,
) -> str:
  huginn_mean_recurrence_arg = (
    "--huginn_mean_recurrence 32" 
    if model_name == "huginn-0125" 
    else ""
  )

  with_intervention_flag = (
    "--with_intervention" 
    if with_intervention 
    else ""
  )

  use_linear_probes_flag = (
    "--use_linear_probes" 
    if with_intervention and use_linear_probes 
    else ""
  )
  linear_probes_file_path_arg = (
    f"--linear_probes_file_path \"$WORKSPACE_PATH/experiments/train_linear_probes/best_checkpoint.pt\"" 
    if with_intervention and use_linear_probes 
    else ""
  )
  
  use_candidate_directions_flag = (
    "--use_candidate_directions" 
    if with_intervention and use_candidate_directions 
    else ""
  )
  candidate_directions_file_path_arg = (
    f"--candidate_directions_file_path \"$WORKSPACE_PATH/experiments/save_candidate_directions/candidate_directions_FIRST_ANSWER_TOKEN.pt\"" 
    if with_intervention and use_candidate_directions 
    else ""
  )

  layer_indices_arg = (
    "--layer_indices " + " ".join(map(str, layer_indices)) 
    if with_intervention and layer_indices 
    else ""
  )
  direction_normalization_mode_arg = (
    f"--direction_normalization_mode {direction_normalization_mode}" 
    if with_intervention and direction_normalization_mode 
    else ""
  )
  projection_hook_mode_arg = (
    f"--projection_hook_mode {projection_hook_mode}" 
    if with_intervention and projection_hook_mode 
    else ""
  )
  modification_mode_arg = (
    f"--modification_mode {modification_mode}"
    if with_intervention and modification_mode
    else ""
  )
  with_hidden_states_pre_hook_flag = (
    "--with_hidden_states_pre_hook" 
    if with_intervention and with_hidden_states_pre_hook 
    else ""
  )
  with_hidden_states_post_hook_flag = (
    "--with_hidden_states_post_hook" 
    if with_intervention and with_hidden_states_post_hook 
    else ""
  )
  scale_arg = (
    f"--scale {scale}" 
    if with_intervention and scale 
    else ""
  )

  num_fewshot_arg = f"--num_fewshot {num_fewshot}"

  return shell.format(
    workspace_path=workspace_path,
    model_name=model_name,

    huginn_mean_recurrence_arg=huginn_mean_recurrence_arg,

    tasks=tasks,

    with_intervention_flag=with_intervention_flag,

    use_linear_probes_flag=use_linear_probes_flag,
    linear_probes_file_path_arg=linear_probes_file_path_arg,
    
    use_candidate_directions_flag=use_candidate_directions_flag,
    candidate_directions_file_path_arg=candidate_directions_file_path_arg,
    
    layer_indices_arg=layer_indices_arg,
    direction_normalization_mode_arg=direction_normalization_mode_arg,
    projection_hook_mode_arg=projection_hook_mode_arg,
    modification_mode_arg=modification_mode_arg,
    with_hidden_states_pre_hook_flag=with_hidden_states_pre_hook_flag,
    with_hidden_states_post_hook_flag=with_hidden_states_post_hook_flag,
    scale_arg=scale_arg,

    num_fewshot_arg=num_fewshot_arg,
  )