def set_up_experiment_jobs(workspace_path: str):
  SAVE_HIDDEN_STATES_CMD = f"""\
  WORKSPACE_PATH={workspace_path}

  python experiments/save_hidden_states/main.py \
  --models_path "$WORKSPACE_PATH/transformers" \
  --model_name huginn-0125 \
  \
  --data_file_path "$WORKSPACE_PATH/datasets/lirefs/mmlu-pro-3000samples.json" \
  --data_name mmlu-pro-3000samples \
  \
  --output_file_path "$WORKSPACE_PATH/experiments/hidden_states_cache/huginn-0125_mmlu-pro-3000samples.pt"
  """

  EVALUATE_REASONING_MEMORIZING_CMD = f"""\
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

  EVALUATE_WITH_LM_EVAL_MMLU_PRO_CMD = f"""\
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
  --tasks mmlu_pro \
  --num_fewshot 5 \
  --batch_size 1 \
  --limit 14 \
  \
  --output_file_path "$WORKSPACE_PATH/experiments/lm_eval_results/huginn-0125.pt"
  """

  EVALUATE_REASONING_MEMORIZING_WITH_INTERVENTION_CMD = f"""\
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
  --layer_indices 66 \
  --with_post_hook \
  \
  --output_file_path "$WORKSPACE_PATH/experiments/reasoning_memorizing_accuracy/huginn-0125.json"
  """

  EVALUATE_WITH_LM_EVAL_MMLU_CMD = f"""\
  WORKSPACE_PATH={workspace_path}

  python experiments/evaluate_lm_eval/main.py \
  --use_local_datasets \
  --data_path "$WORKSPACE_PATH/datasets" \
  \
  --models_path "$WORKSPACE_PATH/transformers" \
  --model_name huginn-0125 \
  \
  --huginn_num_steps 32 \
  \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 4 \
  --limit 50 \
  \
  --output_file_path "$WORKSPACE_PATH/experiments/lm_eval_results/huginn-0125.json"
  """

  EXPERIMENT_JOBS = [
    [
      # save hidden state cache
      SAVE_HIDDEN_STATES_CMD,

      # evaluate reasoning and memorizing accuracy with no intervention
      EVALUATE_REASONING_MEMORIZING_CMD,

      # evaluate with lm-eval with task mmlu_pro
      EVALUATE_WITH_LM_EVAL_MMLU_PRO_CMD,

      # evaluate reasoning and memorizing accuracy with intervention
      EVALUATE_REASONING_MEMORIZING_WITH_INTERVENTION_CMD,
    ],
    [
      # evaluate with lm-eval with task mmlu
      EVALUATE_WITH_LM_EVAL_MMLU_CMD, 
    ]
  ]

  return EXPERIMENT_JOBS