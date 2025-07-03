shell = """\
WORKSPACE_PATH={workspace_path}

python experiments/evaluate_accuracy_reasoning_memorizing/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name {model_name} \
\
{huginn_num_steps} \
\
--test_data_path "$WORKSPACE_PATH/datasets/lirefs" \
--test_data_name mmlu-pro \
--test_data_sample_size 200 \
--with_fewshot_prompts \
--batch_size 1 \
\
{with_intervention_flag} \
{hidden_states_data_file_path} \
{hidden_states_cache_file_path} \
{layer_indices_flag} \
{with_hidden_states_pre_hook_flag} \
{with_hidden_states_post_hook_flag} \
\
--output_file_path "$WORKSPACE_PATH/experiments/reasoning_memorizing_accuracy/{model_name}.json"
"""

def get_mmlu_pro_evaluate_accuracy_reasoning_memorizing(
  workspace_path: str,
  model_name: str,
  with_intervention: bool = False,
  layer_indices: list[int] | None = None,
  with_hidden_states_pre_hook: bool = False,
  with_hidden_states_post_hook: bool = False,
) -> str:
  huginn_num_steps = "--huginn_num_steps 32" if model_name == "huginn-0125" else ""

  with_intervention_flag = "--with_intervention" if with_intervention else ""
  hidden_states_data_file_path = f"--hidden_states_data_file_path \"$WORKSPACE_PATH/datasets/lirefs/mmlu-pro-3000samples.json\"" if with_intervention else ""
  hidden_states_cache_file_path = f"--hidden_states_cache_file_path \"$WORKSPACE_PATH/experiments/hidden_states_cache/{model_name}_mmlu-pro-3000samples.pt\"" if with_intervention else ""
  layer_indices_flag = f"--layer_indices {' '.join(map(str, layer_indices))}" if layer_indices else ""
  with_hidden_states_pre_hook_flag = "--with_hidden_states_pre_hook" if with_hidden_states_pre_hook else ""
  with_hidden_states_post_hook_flag = "--with_hidden_states_post_hook" if with_hidden_states_post_hook else ""

  return shell.format(
    workspace_path=workspace_path,
    model_name=model_name,
    huginn_num_steps=huginn_num_steps,
    with_intervention_flag=with_intervention_flag,
    hidden_states_data_file_path=hidden_states_data_file_path,
    hidden_states_cache_file_path=hidden_states_cache_file_path,
    layer_indices_flag=layer_indices_flag,
    with_hidden_states_pre_hook_flag=with_hidden_states_pre_hook_flag,
    with_hidden_states_post_hook_flag=with_hidden_states_post_hook_flag,
  )