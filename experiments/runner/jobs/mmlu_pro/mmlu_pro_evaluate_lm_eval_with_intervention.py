shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/evaluate_lm_eval/main.py \
--use_local_datasets \
--data_path "$WORKSPACE_PATH/datasets" \
\
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
--device cuda \
--with_parallelize \
\
--huginn_num_steps 32 \
\
--with_intervention \
--hidden_states_cache_file_path "$WORKSPACE_PATH/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt" \
--layer_indices 66 \
--with_post_hook \
\
--tasks mmlu_pro \
--num_fewshot 5 \
--batch_size 1 \
--limit 14 \
\
--output_file_path "$WORKSPACE_PATH/experiments/lm_eval_results/huginn-0125.json"
"""

def get_mmlu_pro_evaluate_lm_eval_with_intervention(workspace_path: str) -> str:
  return shell.format(workspace_path=workspace_path)