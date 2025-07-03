def get_n_layers(model) -> int:
  match model.config.model_type:
    case "huginn_raven":
      return model.config.effective_expected_depth + 1
    case "llama":
      return model.config.num_hidden_layers + 1
    case _:
      raise ValueError(f"Model type {model.config.model_type} is not supported for n_layers retrieval.")