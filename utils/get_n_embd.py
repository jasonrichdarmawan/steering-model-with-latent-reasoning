def get_n_embd(model) -> int:
  match model.config.model_type:
    case "huginn_raven":
      return model.config.n_embd
    case "llama":
      return model.config.hidden_size
    case _:
      raise ValueError(f"Model type {model.config.model_type} is not supported for n_embd retrieval.")