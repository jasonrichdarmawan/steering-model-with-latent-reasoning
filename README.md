# How to save hidden states

```shell
WORKSPACE_PATH="/home/npu-tao/jason"

python experiments/save_hidden_states/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
--data_file_path "$WORKSPACE_PATH/datasets/mmlu-pro-3000samples.json" \
--data_name mmlu-pro-3000samples \
--output_path "$WORKSPACE_PATH/experiments/hidden_states_cache"
```

# How to evaluate accuracy reasoning and memorization

```bash
WORKSPACE_PATH="/home/npu-tao/jason"

python experiments/evaluate_accuracy_reasoning_memorizing/main.py \
--models_path "$WORKSPACE_PATH/transformers" \
--model_name huginn-0125 \
--hidden_states_cache_path "$WORKSPACE_PATH/experiments/hidden_states_cache" \
--mmlu_pro_3000samples_data_file_path "$WORKSPACE_PATH/datasets/mmlu-pro-3000samples.json" \
--test_data_path "$WORKSPACE_PATH/datasets" \
--test_data_name mmlu-pro-3000samples \
--with_fewshot_prompts \
--batch_size 1 \
--with_intervention \
--layer_indices 66 \
--with_post_hook
```

# How to test

```shell
$ python -m unittest discover tests
```

# Disclaimer

- `datasets.zip` is downloaded from [this repository](https://github.com/yihuaihong/Linear_Reasoning_Features/blob/73de7e0802874ad2dc55c1f6aa7d714899fe80f6/dataset.zip)
- `models/recpre` folder is downloaded from [this repository](https://github.com/seal-rg/recurrent-pretraining/tree/9c81784e74b650b06e12d98d23dd7af9aee3571b/recpre). However, the `raven_config_minimal.py` and `raven_modeling_minimal.py` files are downloaded from [this repository](https://huggingface.co/tomg-group-umd/huginn-0125/tree/2a364bd96e3eaa831be324f7c1f9e74892e4e594). This is required because we need `hidden_states` per layer
- `tasks/mmlu` folder is downloaded from [this repository](https://github.com/EleutherAI/lm-evaluation-harness/tree/9fbe48c230c2649d9430c133290d6882b55105ea/lm_eval/tasks/mmlu). This is required because huggingface cannot be accessed directly without a mirror

<details>
<summary>Postponed plan: Minimal code, reproduction of the paper "Understanding Reasoning in Thinking Language Models via Steering Vectors"</summary>

# Note

The authors published the [paper's repository](https://github.com/FlyingPumba/steering-thinking-models) on 25 April 2025. Meanwhile, this repository is specifically for reproducing:
1. how to extract the steering vector candidates
2. how to select the final steering vectors and layers

# To Do

Models used:
- [ ] Thinking models: DeepSeek-R1-Distill-Llama-8B and DeepSeek-R1-Distill-Qwen-14B

  > We conduct our experiments on two DeepSeek-R1-Distill models: Qwen-14B and Llama-8B

- ~~Non-thinking model for comparison, GPT-4o~~

  > This work studies the particular reasoning processes of thinknig LLMs by analyzing DeepSeek-R1-Distill models and comparing them with non-thinking models like GPT-4o

  This is not necessary for steering vector on reasoning model.

- [ ] A model for annotation, GPT-4o

  > To obtain token positions associated with each behavioral category, we generate 300 reasoning chains with the tasks introduced in Section 4.1, using both DeepSeek-R1-Distill models and then annotate them automatically with GPT-4o

- [ ] A model for task generation, Claude 3.5 Sonnet

  > we generate a dataset of 300 tasks across 10 categories using Claude 3.5 Sonnet (see Table 1).

Steps to do:
- [ ] Generate Task Dataset

  > we generate a dataset of 300 tasks across 10 categories using Claude 3.5 Sonnet (see Table 1).

- [ ] Generate Initial Reasoning Chains

  > When generating a reasoning chain, we use greedy decoding and 500 max tokens per response.

- [ ] Annotate Reasoning Chains, identify relevant token positions

  > To obtain token positions asssociated with each behavioral category, we generate 300 reasoning chains with the tasks introduced in Section 4.1, using both DeepSeek-R1-Distill models and then annotate them automatically with GPT-4o.

- [ ] Calculate Layer-wise Candidate Steering Vectors

  See [Core Idea](#core-idea) `To extract a steering vector candidate`

- [ ] Identify Causally Relevant Layers via Attribution Patching

  See [Core Idea](#core-idea) `To select final steering vectors`

  - [ ] Plot these average absolute patching effects per layer (like Figure 3) to identify layers with high causal relevance for each behavior

  - [ ] Select Final Steering Vectors and Layers

- [ ] Select a set of unseen reasoning tasks for evaluating the effects of steering vectors

  > We apply each steering vector to 30 unseen reasoning tasks and analyze how the model's reasoning behavior changes.

- [ ] Evaluate the effects of steering vectors

  See [Core Idea](#core-idea) `When steering in multiple layers`

  > As shown in Figure 4, positive steering increases behaviors such as backtracking, uncertainty estimation, and example testing, while negative steering reduces them.

  - [ ] Quantify Steering Effects (like Figure 4)

# Core idea

- To extract a steering vector candidate 
  > To identify the causally relevant layers for each behavioral category, we first extract a steering vector candidate from every layer using the Difference of Means method (Section 2.2):
  > $$u_\mathcal{l}^c = \frac{1}{ | D_+ | } \sum_{ p_i \in D_+ }{\overline{a}_\mathcal{l}^c(p_i)} - \frac{1}{ | D_- | } \sum_{p_j \in D_-}{ a_\mathcal{l}^c(p_j) }, \quad \text{with} \quad \overline{a}_\mathcal{l}^c(p_i) = \frac{1}{ | \text{seq}_c(p_i) | } \sum_{t \in \text{seq}_c(p_i) }{ a_\mathcal{l}(t) }$$
  > where:
  > - $a_\mathcal{l}(t)$ represents the residual stream activation at layer $\mathcal{l}$ for token position $t$.
  > $\text{seq}_c(p)$ is the set of all token sequences within the prompt $$p$$ that are annotated with category $c$, including the preceding token position.
  > $\overline{a}_\mathcal{l}^c(p_i)$ denotes the mean activation across all token positions within the annotated sequences of category $c$ at layer $\mathcal{l}$
  > - $D_+$ consists of prompts containing at least one sequence labeled with category c, while $D_-$ represents the full dataset.
  >
  > The resulting vector $u_\mathcal{l}^c$ serves as a candidate steering vector for each layer.

- To select final steering vectors

  > To determine the final steerign vectors, we apply attribution patching (Section 2.1) to quantify the causal relevance of each vector in its respective layer.
  > Specifically, we consider the following patching experiment: Given a candidate steering vector $u_\mathcal{l}^c$ for a specific behavioral category, we add it to the residual stream activation preceding a token-sequence annotated with one of the relevant behaviors. Therefore, we define the patched activation as:
  > $$a_\mathcal{l}^\text{patched} = a_\mathcal{l} + u_\mathcal{l}^c$$
  > If this intervention leads to a significant change in the KL divergence of the next-t0ken prediction, then the steerign vector in layer $\mathcal{l}$ is causally relevant for the given behavior. We approximate the patching effect for this experiment with:
  > $$\Delta L \approx (u_\mathcal{l}^c)^T \cdot \frac{ \delta }{ \delta a_\mathcal{l} } L \left( \begin{array}{c|c} X_\text{clean} & \text{do}(a_\mathcal{l} = a_\text{clean} ) \end{array} \right) $$
  > where $u_\mathcal{l}^c = ( a_\mathcal{l}^\text{patched} - a_\mathcal{l} )$.
  > We average the absolute patching effect for each category over all category-sequences in all 300 reasoning chains.

- When steering in multiple layers

  > To evalaute the effectiveness of our extracted steering vectors, we apply them at the selected layers, we apply them at the selected layers (see Table 2) and observe their influence on the model's reasoning process.
  > Steering is implemented by adding or subtracting the extracted steering vectors $u_\mathcal{l}^c$ to the residual stream activations at inference time.
  > When steering in multiple layers simultaneously, we scale each addition or subtraction b a coefficient equal to the reciprocal of the number of layers.
  > By applying this intervention, we can increase or decrease behaviors such as backtracking, uncertainty estimation, and example testing, providing a direct mechanism for manipulating the model's reasoning process.
  </details>