from os.path import join

import torch
from transformers import (
  AutoModelForCausalLM, 
  AutoTokenizer, 
  PreTrainedTokenizerFast
)

# Use custom code for Huginn models
from models import recpre

def load_model_and_tokenizer(
  models_path: str, 
  model_name: str,
  device_map: str | dict[str, int] = "cuda" if torch.cuda.is_available() else "cpu",
  torch_dtype: torch.dtype | None = None,
):
  model = load_model(
    models_path=models_path, 
    model_name=model_name,
    device_map=device_map,
    torch_dtype=torch_dtype,
  )

  tokenizer: PreTrainedTokenizerFast = load_tokenizer(
    models_path=models_path,
    model_name=model_name,
  )

  return model, tokenizer

def load_model(
  models_path: str, 
  model_name: str, 
  device_map: str,
  torch_dtype: torch.dtype | None = None,
):
  trust_remote_code = True

  # Restore the default settings for torch.backends.cuda
  # to avoid RuntimeError: No available kernel. Aborting execution.
  # The error was caused because we import the models.recpre folder
  torch.backends.cuda.enable_flash_sdp(True)
  torch.backends.cuda.enable_math_sdp(True)
  torch.backends.cuda.enable_mem_efficient_sdp(True)
  torch.backends.cuda.enable_cudnn_sdp(True)

  match model_name:
    case "huginn-0125":
      # Reference: https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
      if torch_dtype is None:
        torch_dtype = torch.bfloat16

      # Use custom code for Huginn models
      trust_remote_code = False
    case "Meta-Llama-3-8B":
      # Reference: https://huggingface.co/meta-llama/Meta-Llama-3-8B
      if torch_dtype is None:
        torch_dtype = torch.bfloat16
    case _:
      raise ValueError(f"Unsupported model name: {model_name}")

  # device_map is important because Huginn model
  # use torch_dtype=torch.bfloat16
  model = AutoModelForCausalLM.from_pretrained(
    join(models_path, model_name),
    torch_dtype=torch_dtype,
    trust_remote_code=trust_remote_code,
    device_map=device_map,
  )

  return model

def load_tokenizer(
  models_path: str, 
  model_name: str,
):
  tokenizer = AutoTokenizer.from_pretrained(
    join(models_path, model_name),
    trust_remote_code=True,
  )

  match model_name:
    case "huginn-0125":
      """
      Reference: https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
      """
      print("Using Huginn tokenizer settings.")
      tokenizer.eos_token_id = 65505
      tokenizer.bos_token_id = 65504
      tokenizer.pad_token_id = 65509
    case "Meta-Llama-3-8B":
      tokenizer.pad_token_id = tokenizer.eos_token_id
    case _:
      raise ValueError(f"Unsupported model name: {model_name}")

  # Left-padding is more straightforward to get
  # the last token in the batch.
  tokenizer.padding_side = "left"

  return tokenizer