from utils import ProcessHiddenStatesMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/save_candidate_directions/main.py \
{models_name_arg} \
\
{hidden_states_file_paths_arg} \
\
{process_hidden_states_mode_arg} \
\
--output_path "$WORKSPACE_PATH/experiments/save_candidate_directions"
"""

def handle_save_candidate_directions(
  workspace_path: str,
  job: str,
):
  if job == "save_candidate_directions":
    return get_save_candidate_directions(
      workspace_path=workspace_path,
      models_name=["huginn-0125"],
    )
  elif job == "save_candidate_directions_model_name_Meta-Llama-3-8B":
    return get_save_candidate_directions(
      workspace_path=workspace_path,
      models_name=["Meta-Llama-3-8B"],
    )

def get_save_candidate_directions(
  workspace_path: str,
  models_name: list[str],
) -> str:
  models_name_arg = f"--models_name " + " ".join(models_name)

  hidden_states_file_paths = [
    f"{workspace_path}/experiments/save_hidden_states/{model_name}_hidden_states_FIRST_ANSWER_TOKEN.pt"
    for model_name in models_name
  ]
  hidden_states_file_paths_args = f"--hidden_states_file_paths " + " ".join(hidden_states_file_paths)

  return shell.format(
    workspace_path=workspace_path,
    models_name_arg=models_name_arg,
    hidden_states_file_paths_arg=hidden_states_file_paths_args,
  )