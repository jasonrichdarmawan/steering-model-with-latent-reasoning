shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/analyze_steering_effect_per_layer/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
\
--hidden_states_cache_file_path "$WORKSPACE_PATH/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt" \
\
--data_path "$WORKSPACE_PATH/datasets/lirefs" \
--data_name mmlu-pro-3000samples.json \
--data_batch_size 1 \
\
--huginn_num_steps 16 \
\
--with_post_hook \
--projection_scale 0.1 \
\
--output_file_path "$WORKSPACE_PATH/experiments/analyze_steering_effect_per_layer/huginn-0125.json"
"""

def get_mmlu_pro_analyze_steering_effect_per_layer(workspace_path: str) -> str:
    return shell.format(workspace_path=workspace_path)