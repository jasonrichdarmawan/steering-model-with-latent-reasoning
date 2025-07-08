from utils import ProcessHiddenStatesMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/save_candidate_directions/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
\
{huginn_num_steps_flag} \
\
--data_path "$WORKSPACE_PATH/datasets/lirefs" \
--data_name mmlu-pro-3000samples.json \
\
--data_batch_size 1 \
\
{process_hidden_states_mode_arg} \
\
--output_path "$WORKSPACE_PATH/experiments/save_candidate_directions"
"""

def get_save_candidate_directions(
  workspace_path: str,
  model_name: str,
) -> str:
  huginn_num_steps_flag = "--huginn_num_steps 32" if model_name == "huginn-0125" else ""

  process_hidden_states_mode_arg = f"--process_hidden_states_mode {str(ProcessHiddenStatesMode.FIRST_ANSWER_TOKEN)}"

  return shell.format(
    workspace_path=workspace_path,
    model_name=model_name,
    huginn_num_steps_flag=huginn_num_steps_flag,
    process_hidden_states_mode_arg=process_hidden_states_mode_arg,
  )