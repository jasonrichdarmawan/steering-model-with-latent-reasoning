from utils import DirectionNormalizationMode
from utils import ProcessHiddenStatesMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/analyze_steering_effect_per_layer/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
\
--huginn_num_steps 32 \
\
{process_hidden_states_mode_arg} \
{candidate_directions_file_path_arg} \
{direction_normalization_flag} \
\
{data_path_arg} \
{data_name_arg} \
--data_batch_size 1 \
\
--output_path "$WORKSPACE_PATH/experiments/analyze_steering_effect_per_layer/huginn-0125"
"""

def get_analyze_steering_effect_per_layer(
  workspace_path: str,
  model_name: str,
  data_path: str,
  data_name: str,
) -> str:
  direction_normalization = DirectionNormalizationMode.UNIT_VECTOR
  direction_normalization_flag = f"--direction_normalization_mode {direction_normalization}"

  data_path_arg = f"--data_path \"$WORKSPACE_PATH/{data_path}\""
  data_name_arg = f"--data_name {data_name}"

  process_hidden_states_mode = ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN
  process_hidden_states_mode_arg = f"--process_hidden_states_mode {process_hidden_states_mode}"

  candidate_directions_file_path_arg = f"--candidate_directions_file_path \"$WORKSPACE_PATH/experiments/save_candidate_directions/{model_name}_mmlu-pro-3000samples.json_{process_hidden_states_mode}_candidate_directions.pt\""

  return shell.format(
    workspace_path=workspace_path,

    model_name=model_name,

    data_path_arg=data_path_arg,
    data_name_arg=data_name_arg,

    process_hidden_states_mode_arg=process_hidden_states_mode_arg,
    candidate_directions_file_path_arg=candidate_directions_file_path_arg,
    direction_normalization_flag=direction_normalization_flag,
  )