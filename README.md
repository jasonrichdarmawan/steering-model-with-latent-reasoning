# Installation

1. Install the Python project's dependencies

   ```bash
   conda create -n recurrent-env python=3.11
   conda activate recurrent-env
   pip install -r requirements.txt
   ```

2. Use the following folder structure

   - [ ] Support the default huggingface's folder: `~/.cache/huggingface`

         It is not supported natively because huggingface.co is not accesible from the author's location, making automated download not feasible. For now, please download the model's weights and datasets from huggingface using `huggingface-cli`.

         Note: If the datasets have Python files, you need to manually download it as `huggingface-cli` will only not download it.

   ```
   WORKSPACE/
   |- PROJECT/
   |- transformers/
   |- datasets/
   |- experiments/
   ```

   The `$WORKSPACE/transformers` folder is where you store the model's weights. For example, `$WORKSPACE/transformers/huginn-0125/model-00003-of-00004.safetensors`.

   The `$WORKSPACE/datasets` folder is where you store the datasets, with the following format `$WORKSPACE/<user-name>/<dataset-name>`. For example, `$WORKSPACE/cais/mmlu`.

3. Manually download [datasets.zip](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/73de7e0802874ad2dc55c1f6aa7d714899fe80f6/dataset.zip)

   - [ ] Upload the `datasets.zip` to GitHub LFS for future proofing

   Extract the `datasets.zip` to `$WORKSPACE/datasets/lirefs`.

   Note: We use `datasets.zip` for fair performance comparison purpose as [the method](https://arxiv.org/abs/2503.23084) we are comparing to use this datasets.

# How to reproduce the experiment?

See the subsections below for the examples.

You can see the list of jobs in the [runner/jobs](runner/jobs) folder.

A job consists of multiple sub-jobs. If you want to run specific
sub-job, then see the `runner/jobs/[job_name]/_[job_name].py` file

## How to save hidden states?

```shell
WORKSPACE_PATH="/media/npu-tao/disk4T/jason"

python runner/main.py \
--workspace_path "$WORKSPACE_PATH" \
--jobs save_hidden_states \
--output_path "$WORKSPACE_PATH/experiments/runner"
```

Jobs:
- save_hidden_states

## How to save candidate directions?

```shell
WORKSPACE_PATH="/root/autodl-fs"

python runner/main.py \
--workspace_path "$WORKSPACE_PATH" \
--jobs mmlu_pro_save_candidate_directions_all_tokens \
--output_path "$WORKSPACE_PATH/experiments/runner"
```

Jobs:
- mmlu_pro_save_candidate_directions
- mmlu_pro_save_candidate_directions_all_tokens
- mmlu_pro_meta-llama-3-8b_save_candidate_directions
- mmlu_pro_meta-llama-3-8b_save_candidate_directions_all_tokens

Note:
- mmlu_pro_save_candidate_directions_all_tokens with batch size 2 requires 1x 4090 or equivalent

## How to analyze steering effect per layer?

```bash
WORKSPACE_PATH="/media/npu-tao/disk4T/jason"

python runner/main.py \
--workspace_path "$WORKSPACE_PATH" \
--jobs mmlu_pro_analyze_steering_effect_per_layer \
--output_path "$WORKSPACE_PATH/experiments/runner"
```

Jobs:
- mmlu_pro_analyze_steering_effect_per_layer
- mmlu_pro_analyze_steering_effect_per_layer_all_tokens
- mmlu_pro_meta-llama-3-8b_analyze_steering_effect_per_layer
- mmlu_pro_meta-llama-3-8b_analyze_steering_effect_per_layer_all_tokens

Note:
- mmlu_pro_analyze_steering_effect_per_layer and its derivatives with batch size 1 require 4x 3090 or equivalent

## How to evaluate accuracy reasoning and memorization?

```bash
WORKSPACE_PATH="/media/npu-tao/disk4T/jason"

python runner/main.py \
--workspace_path "$WORKSPACE_PATH" \
--jobs mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_1 \
--output_path "$WORKSPACE_PATH/experiments/runner"
```

Tasks:
- mmlu_pro_evaluate_accuracy_reasoning_memorizing
- mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention
- mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_129
- mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_1
- mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention_1_all_tokens
- mmlu_pro_meta-llama-3-8b_evaluate_accuracy_reasoning_memorizing
- mmlu_pro_meta-llama-3-8b_evaluate_accuracy_reasoning_memorizing_with_intervention

Note:
- mmlu_pro_evaluate_accuracy_reasoning_memorizing and its derivatives with batch size 1 require 2x 3090 or equivalent

## How to evaluate with lm_eval?

```bash
WORKSPACE_PATH="/media/npu-tao/disk4T/jason"

python runner/main.py \
--workspace_path "$WORKSPACE_PATH" \
--jobs mmlu_evaluate_lm_eval_with_intervention_use_linear_probes_few_shots_1 \
--output_path "$WORKSPACE_PATH/experiments/runner" \
--shutdown_after_experiment
```

Jobs:
- mmlu_pro_evaluate_lm_eval
- mmlu_pro_evaluate_lm_eval_few_shots_1
- mmlu_pro_evaluate_lm_eval_with_intervention
- mmlu_pro_evaluate_lm_eval_with_intervention_129
- mmlu_evaluate_lm_eval
- mmlu_evaluate_lm_eval_with_intervention
- mmlu_evaluate_lm_eval_with_intervention_scale_with_overall_magnitude_feature_amplification
- mmlu_evaluate_lm_eval_with_intervention_1
- mmlu_evaluate_lm_eval_with_intervention_127
- mmlu_evaluate_lm_eval_with_intervention_129
- mmlu_evaluate_lm_eval_with_intervention_1_all_tokens
- mmlu_evaluate_lm_eval_with_intervention_1_all_tokens_scale_with_overall_magnitude
- mmlu_evaluate_lm_eval_with_intervention_use_linear_probes
- mmlu_evaluate_lm_eval_with_intervention_use_linear_probes_few_shots_1

Note:
- mmlu_pro_evaluate_lm_eval and its derivatives with batch size 4 require 2x 3060 or equivalent

## How to test?

```shell
$ python -m unittest discover tests
```

# Disclaimer

- `datasets.zip` is downloaded from [this repository](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/73de7e0802874ad2dc55c1f6aa7d714899fe80f6/dataset.zip)
- `models/recpre` folder is downloaded from [this repository](https://github.com/seal-rg/recurrent-pretraining/tree/9c81784e74b650b06e12d98d23dd7af9aee3571b/recpre). However, the `raven_config_minimal.py` and `raven_modeling_minimal.py` files are downloaded from [this repository](https://huggingface.co/tomg-group-umd/huginn-0125/tree/2a364bd96e3eaa831be324f7c1f9e74892e4e594). This is required because we need `hidden_states` per layer

# Hardware used

2x NVIDIA 3060 was used for the following jobs:
- `mmlu_evaluate_lm_eval`
- `mmlu_evaluate_lm_eval_with_intervention`

1x NVIDIA 3090 was used for the following jobs:
- `mmlu_pro_save_hidden_states`

2x NVIDIA 3090 was used for the following jobs:
- `mmlu_pro_evaluate_accuracy_reasoning_memorizing`
- `mmlu_pro_evaluate_accuracy_reasoning_memorizing_with_intervention`
- `mmlu_pro_evaluate_lm_eval`
- `mmlu_pro_evaluate_lm_eval_with_intervention`

4x NVIDIA 3090 was used for the following jobs:
- `mmlu_pro_analyze_steering_effect_per_layer`
