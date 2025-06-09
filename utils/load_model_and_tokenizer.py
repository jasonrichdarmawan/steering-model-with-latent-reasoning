from os.path import join

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use custom code for Huginn models
import recpre # type: ignore

def load_model_and_tokenizer(model_path: str, 
                             model_name: str):
  model = load_model(model_path, 
                     model_name)

  tokenizer = load_tokenizer(model_path, 
                             model_name)

  model.to("cuda")

  return model, tokenizer

def load_model(model_path: str, model_name: str):

  torch_dtype = torch.float32
  trust_remote_code = True
  
  match model_name:
    case name if name.startswith("huginn-"):
      # Reference: https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
      torch_dtype = torch.bfloat16

      # Use custom code for Huginn models
      trust_remote_code = False

  model = AutoModelForCausalLM.from_pretrained(
    join(model_path, model_name),
    torch_dtype=torch_dtype,
    trust_remote_code=trust_remote_code,
  )

  return model

def load_tokenizer(model_path: str, 
                   model_name: str):
  tokenizer = AutoTokenizer.from_pretrained(
    join(model_path, model_name),
    trust_remote_code=True,
  )

  match model_name:
    case name if "huginn" in name:
      """
      Reference: https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
      """
      print("Using Huginn tokenizer settings.")
      tokenizer.eos_token_id = 65505
      tokenizer.bos_token_id = 65504
      tokenizer.pad_token_id = 65509

  # Left-padding is more straightforward to get
  # the last token in the batch.
  tokenizer.padding_side = "left"

  return tokenizer