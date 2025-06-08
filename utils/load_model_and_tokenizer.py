from os.path import join

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path: str, 
                             model_name: str):
  model = AutoModelForCausalLM.from_pretrained(
    join(model_path, model_name),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(
    join(model_path, model_name),
    trust_remote_code=True,
  )

  return model, tokenizer