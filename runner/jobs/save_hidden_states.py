from utils import CacheHiddenStatesMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/save_hidden_states/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
\
--data_path "$WORKSPACE_PATH/datasets/lirefs" \
--data_name mmlu-pro-3000samples.json \
\
--data_batch_size 1 \
\
{cache_hidden_states_mode_arg} \
\
--output_path "$WORKSPACE_PATH/experiments/save_hidden_states"
"""

def handle_save_hidden_states(
  workspace_path: str,
  job: str,
):
  if job == "save_hidden_states":
    return get_save_hidden_states(
      workspace_path=workspace_path,
      model_name="huginn-0125",
      cache_hidden_states_mode=CacheHiddenStatesMode.FIRST_ANSWER_TOKEN,
    )
  elif job == "save_hidden_states_model_name_Meta-Llama-3-8B":
    return get_save_hidden_states(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
      cache_hidden_states_mode=CacheHiddenStatesMode.FIRST_ANSWER_TOKEN,
    )

def get_save_hidden_states(
  workspace_path: str,
  model_name: str,
  cache_hidden_states_mode: CacheHiddenStatesMode,
) -> str:
  cache_hidden_states_mode_arg = f"--cache_hidden_states_mode {cache_hidden_states_mode}"

  return shell.format(
    workspace_path=workspace_path,
    model_name=model_name,
    cache_hidden_states_mode_arg=cache_hidden_states_mode_arg,
  )