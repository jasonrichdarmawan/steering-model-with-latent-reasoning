shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/save_hidden_states/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
\
--data_path "$WORKSPACE_PATH/datasets/lirefs" \
--data_name mmlu-pro-3000samples.json \
\
\
--output_file_path "$WORKSPACE_PATH/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt"
"""

def get_mmlu_pro_save_hidden_states(workspace_path: str) -> str:
    return shell.format(workspace_path=workspace_path)