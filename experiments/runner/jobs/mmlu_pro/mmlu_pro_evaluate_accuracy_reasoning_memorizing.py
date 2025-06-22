shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/evaluate_accuracy_reasoning_memorizing/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
--device auto \
\
--huginn_num_steps 32 \
\
--test_data_path "$WORKSPACE_PATH/datasets/lirefs" \
--test_data_name mmlu-pro-3000samples.json \
--with_fewshot_prompts \
--batch_size 1 \
\
--output_file_path "$WORKSPACE_PATH/experiments/reasoning_memorizing_accuracy/huginn-0125.json"
"""

def get_mmlu_pro_evaluate_accuracy_reasoning_memorizing(workspace_path: str) -> str:
    return shell.format(workspace_path=workspace_path)