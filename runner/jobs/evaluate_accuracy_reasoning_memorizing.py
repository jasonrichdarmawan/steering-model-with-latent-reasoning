from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/evaluate_accuracy_reasoning_memorizing/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
\
{huginn_num_steps} \
\
{test_data_path_arg} \
{test_data_name_arg} \
--with_fewshot_prompts \
--batch_size 1 \
\
{with_intervention_flag} \
{candidate_directions_file_path_arg} \
{direction_normalization_mode_arg} \
{modification_mode_arg} \
{projection_hook_mode_arg} \
{layer_indices_arg} \
{with_hidden_states_pre_hook_flag} \
{with_hidden_states_post_hook_flag} \
{scale_arg} \
\
--output_file_path "$WORKSPACE_PATH/experiments/reasoning_memorizing_accuracy/{model_name}.json"
"""

def handle_evaluate_accuracy_reasoning_memorizing(
  workspace_path: str,
  job: str,
) -> str:
  if job == "evaluate_accuracy_reasoning_memorizing_model_name_Meta-Llama-3-8B":
    return get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro-3000samples.json",
      with_intervention=False,
    )
  elif job == "evaluate_accuracy_reasoning_memorizing_model_name_Meta-Llama-3-8B_with_intervention_layer_indices_8_scale_-5e-2":
    return get_evaluate_accuracy_reasoning_memorizing(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      test_data_path="datasets/lirefs",
      test_data_name="mmlu-pro-3000samples.json",
      with_intervention=True,

      layer_indices=[8],
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
      projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
      modification_mode=TokenModificationMode.ALL_TOKENS,
      with_hidden_states_pre_hook=True,
      with_hidden_states_post_hook=False,
      scale=-5e-2,
    )

def get_evaluate_accuracy_reasoning_memorizing(
  workspace_path: str,
  model_name: str,
  test_data_path: str,
  test_data_name: str,
  with_intervention: bool,

  layer_indices: list[int] | None = None,
  direction_normalization_mode: DirectionNormalizationMode | None = None,
  projection_hook_mode: ProjectionHookMode | None = None,
  modification_mode: TokenModificationMode | None = None,
  with_hidden_states_pre_hook: bool = False,
  with_hidden_states_post_hook: bool = False,
  scale: float | None = None,
) -> str:
  huginn_num_steps = (
    "--huginn_num_steps 32" 
    if model_name == "huginn-0125" 
    else ""
  )

  test_data_path_arg = f"--test_data_path \"$WORKSPACE_PATH/{test_data_path}\""
  test_data_name_arg = f"--test_data_name {test_data_name}"

  with_intervention_flag = (
    "--with_intervention"
    if with_intervention 
    else ""
  )

  candidate_directions_file_path_arg = (
    f"--candidate_directions_file_path \"$WORKSPACE_PATH/experiments/save_candidate_directions/candidate_directions_FIRST_ANSWER_TOKEN.pt\"" 
    if with_intervention 
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
  
  layer_indices_arg = (
    f"--layer_indices {' '.join(map(str, layer_indices))}" 
    if with_intervention and layer_indices 
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

  return shell.format(
    workspace_path=workspace_path,
    
    model_name=model_name,
    
    test_data_path_arg=test_data_path_arg,
    test_data_name_arg=test_data_name_arg,

    huginn_num_steps=huginn_num_steps,
    
    with_intervention_flag=with_intervention_flag,
    candidate_directions_file_path_arg=candidate_directions_file_path_arg,
    direction_normalization_mode_arg=direction_normalization_mode_arg,
    projection_hook_mode_arg=projection_hook_mode_arg,
    modification_mode_arg=modification_mode_arg,
    layer_indices_arg=layer_indices_arg,
    with_hidden_states_pre_hook_flag=with_hidden_states_pre_hook_flag,
    with_hidden_states_post_hook_flag=with_hidden_states_post_hook_flag,
    scale_arg=scale_arg,
  )