import torch
from torch import nn
from transformers import GenerationConfig, PreTrainedTokenizerBase

def generate_sentences(
  model: nn.Module,
  tokenizer: PreTrainedTokenizerBase,
  text: list[str],
  config: GenerationConfig,
):
  match model.config.model_type:
    case name if "huginn_" in name:
      """
      Reference: 
      [1] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
      [2] https://github.com/seal-rg/recurrent-pretraining/blob/9f84159bc548f4fe75a577d71575c35ef80e1977/examples/inference_demo.ipynb
      """
      input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        padding="longest",
        return_token_type_ids=False,
      ).input_ids.to("cuda")

      with torch.no_grad():
        outputs = model.generate(
          input_ids,
          config,
          num_steps=32,
          tokenizer=tokenizer
        )

      generated_texts = tokenizer.batch_decode(
        outputs.sequences[:, input_ids.shape[1]:],
        skip_special_tokens=True,
      )
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")
    
  return generated_texts