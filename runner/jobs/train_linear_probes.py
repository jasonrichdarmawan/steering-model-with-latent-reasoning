from utils import CacheHiddenStatesMode

shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/train_linear_probes/main.py \
--model_name {model_name} \
--device cuda:0 \
\
--hidden_states_file_path "{WORKSPACE_PATH}/experiments/save_hidden_states/{model_name}_hidden_states_FIRST_ANSWER_TOKEN.pt" \
\
--epochs 1000 \
\
--lr 0.00005 \
--weight_decay 0.01 \
\
--test_ratio 0.25 \
--batch_size 32 \
\
--output_dir "$WORKSPACE_PATH/experiments/train_linear_probes/{model_name}" \
--checkpoint_freq 50 \
--use_early_stopping \
--early_stopping_patience 10
"""

def handle_train_linear_probes(
  workspace_path: str,
  job: str,
):
  if job == "train_linear_probes_model_name_huginn-0125":
    return get_train_linear_probes(
      workspace_path=workspace_path,
      model_name="huginn-0125",
    )
  elif job == "train_linear_probes_model_name_Meta-Llama-3-8B":
    return get_train_linear_probes(
      workspace_path=workspace_path,
      model_name="Meta-Llama-3-8B",
    )
  else:
    raise ValueError(f"Unknown job: {job}")

def get_train_linear_probes(
  workspace_path: str,
  model_name: str,
) -> str:
  return shell.format(
    workspace_path=workspace_path,
    model_name=model_name,
  )