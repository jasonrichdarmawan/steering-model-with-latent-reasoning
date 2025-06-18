import torch
from torch import nn
from transformers import GenerationConfig, PreTrainedTokenizerBase

def generate_sentences(
  model: nn.Module,
  tokenizer: PreTrainedTokenizerBase,
  text: list[str],
):
  match model.config.model_type:
    case name if "huginn_" in name:
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
        pad_token_id=65509
      )

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
          tokenizer=tokenizer,
        )

      generated_texts = tokenizer.batch_decode(
        outputs[:, input_ids.shape[1]:],
        skip_special_tokens=True,
      )

      return generated_texts
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")