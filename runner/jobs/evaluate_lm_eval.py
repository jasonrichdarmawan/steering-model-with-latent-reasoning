from utils import ProcessHiddenStatesMode
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

def get_evaluate_lm_eval(
  workspace_path: str,
  model_name: str,
  tasks: str,
  with_intervention: bool,
  num_fewshot: int = 0,
  
  use_linear_probes: bool = False,

  use_candidate_directions: bool = False,
  process_hidden_states_mode: ProcessHiddenStatesMode | None = None,

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
  process_hidden_states_mode_arg = (
    f"--process_hidden_states_mode {process_hidden_states_mode}" 
    if with_intervention and use_candidate_directions
    and process_hidden_states_mode
    else ""
  )
  candidate_directions_file_path_arg = (
    f"--candidate_directions_file_path \"$WORKSPACE_PATH/experiments/save_candidate_directions/{model_name}_mmlu-pro-3000samples.json_{process_hidden_states_mode}_candidate_directions.pt\"" 
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
    process_hidden_states_mode_arg=process_hidden_states_mode_arg,
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