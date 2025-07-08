from transformers import PreTrainedTokenizerBase

def tokenize_text(
  model,
  tokenizer: PreTrainedTokenizerBase,
  text: list[str],
):
  """
  model.config.model_type:
    - "huginn_raven": Make sure to use torch.nn.functional.scaled_dot_product_attention,
    which expects `True` for positions to mask out.
    Make sure to use `return_head=False` in the model's forward method.
    Otherwise, change the attention_mask to `... = 0` because
    self.compute_eager_sdpa use `torch.masked_fill`, which
    expects `True` for positions to keep.
  """
  inputs = tokenizer(
    text=text,
    return_tensors="pt",
    add_special_tokens=True,
    padding="longest",
    return_token_type_ids=False,
  ).to(model.device)

  return inputs