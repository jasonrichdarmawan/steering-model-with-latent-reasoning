shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/evaluate_accuracy_reasoning_memorizing/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
--device auto \
\
--huginn_num_steps 32 \
\
--hidden_states_cache_file_path "$WORKSPACE_PATH/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt" \
\
--test_data_path "$WORKSPACE_PATH/datasets/lirefs" \
--test_data_name mmlu-pro-3000samples.json \
--with_fewshot_prompts \
--batch_size 1 \
\
--with_intervention \
--layer_indices {layer_indices} \
--with_post_hook \
\
--output_file_path "$WORKSPACE_PATH/experiments/reasoning_memorizing_accuracy/huginn-0125.json"
"""

def get_mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention(
  workspace_path: str,
  layer_indices: list[int],
) -> str:
  layer_indices_str = " ".join(map(str, layer_indices))

  return shell.format(
    workspace_path=workspace_path,
    layer_indices=layer_indices_str,
  )