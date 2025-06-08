from os.path import join

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path: str, 
                             model_name: str):
  model = AutoModelForCausalLM.from_pretrained(
    join(model_path, model_name),
    torch_dtype=torch.float32,
    trust_remote_code=True,
  )

  tokenizer = load_tokenizer(model_path, model_name)

  model.to("cuda")

  return model, tokenizer

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

  tokenizer.padding_side = "left"

  return tokenizer