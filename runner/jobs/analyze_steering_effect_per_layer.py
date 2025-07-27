from utils import DirectionNormalizationMode
from utils import ProcessHiddenStatesMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/analyze_steering_effect_per_layer/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
\
{huginn_num_steps_flag} \
\
{candidate_directions_file_path_arg} \
{direction_normalization_mode_flag} \
\
{data_path_arg} \
{data_name_arg} \
--data_batch_size 2 \
\
--output_path "$WORKSPACE_PATH/experiments/analyze_steering_effect_per_layer/huginn-0125"
"""

def handle_analyze_steering_effect_per_layer(
  workspace_path: str,
  job: str,
) -> str:
  if job == "analyze_steering_effect_per_layer_model_name_huginn-0125":
    return get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
    )
  elif job == "analyze_steering_effect_per_layer_model_name_Meta-Llama-3-8B":
    return get_analyze_steering_effect_per_layer(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      data_path="datasets/lirefs",
      data_name="mmlu-pro-3000samples.json",
      direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
    )
  else:
    raise ValueError(f"Job {job} is not recognized for analyze steering effect per layer.")

def get_analyze_steering_effect_per_layer(
  workspace_path: str,
  model_name: str,
  data_path: str,
  data_name: str,
  direction_normalization_mode: DirectionNormalizationMode,
) -> str:
  huginn_num_steps_flag = "--huginn_num_steps 32" if model_name == "huginn-0125" else ""

  direction_normalization_mode_flag = f"--direction_normalization_mode {direction_normalization_mode}"

  data_path_arg = f"--data_path \"$WORKSPACE_PATH/{data_path}\""
  data_name_arg = f"--data_name {data_name}"

  candidate_directions_file_path_arg = f"--candidate_directions_file_path \"$WORKSPACE_PATH/experiments/save_candidate_directions/candidate_directions_FIRST_ANSWER_TOKEN.pt\""

  return shell.format(
    workspace_path=workspace_path,

    model_name=model_name,

    huginn_num_steps_flag=huginn_num_steps_flag,

    data_path_arg=data_path_arg,
    data_name_arg=data_name_arg,

    candidate_directions_file_path_arg=candidate_directions_file_path_arg,
    direction_normalization_mode_flag=direction_normalization_mode_flag,
  )