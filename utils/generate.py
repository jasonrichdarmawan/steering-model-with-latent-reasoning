import torch
from torch import nn
from torch import Tensor
from jaxtyping import Float
from transformers import GenerationConfig
from transformers import PreTrainedTokenizerBase

def generate(
  model: nn.Module,
  input_ids: Float[Tensor, "batch seq_len"],
  attention_mask: Float[Tensor, "batch seq_len"],
  tokenizer: PreTrainedTokenizerBase | None = None,
  huginn_num_steps: int | None = None,
):
  match model.config.model_type:
    case "huginn_raven":
      return generate_huginn(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_steps=huginn_num_steps,
      )
    case "llama":
      return generate_lirefs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
    case _:
      raise ValueError(
        f"Unsupported model type: {model.config.model_type}. "
        "Supported types are: 'huginn_raven', 'llama'."
      )

def generate_huginn(
  model: nn.Module,
  tokenizer: PreTrainedTokenizerBase,
  input_ids: Float[Tensor, "batch seq_len"],
  attention_mask: Float[Tensor, "batch seq_len"],
  num_steps: int | None = None
):
  """
  Reference: 
  [1] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  [2] https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
  [3] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/main/reasoning_representation/Intervention/features_intervention.py
  [4] https://github.com/seal-rg/recurrent-pretraining/blob/0d9ed974d253e16498edec5c0c0916fdef4eb339/evaluate_raven/hf_eval_adaptive_compute.py
  """
  config = GenerationConfig(
    max_new_tokens=200,
    stop_strings=["<|end_text|>", "<|end_turn|>"],
    do_sample=False,
    temperature=None,
    top_p=None,
    min_p=None,
    return_dict_in_generate=False,
    eos_token_id=65505,
    bos_token_id=65504,
    pad_token_id=65509,
  )

  outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=config,
    num_steps=num_steps,
    tokenizer=tokenizer,
  )

  return outputs

def generate_lirefs(
  model: nn.Module,
  input_ids: Float[Tensor, "batch seq_len"],
  attention_mask: Float[Tensor, "batch seq_len"],
):
  outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    do_sample=False,
  )

  return outputs