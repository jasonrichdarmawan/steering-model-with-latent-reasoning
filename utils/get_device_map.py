from utils import get_n_layers

def get_device_map(
  model,
) -> dict[int, int]:
  device_map: dict[int, int] = {}

  match model.config.model_type:
    case "huginn_raven":
      n_layers = get_n_layers(model)

      if len(model.hf_device_map) == 1:
        # If the model is on a single device, assign all layers to that device
        device = model.hf_device_map['']
        for i in range(n_layers):
          device_map[i] = device
        return device_map

      recurrent_block_end_index = (
        model.config.n_layers_in_prelude
        + model.config.n_layers_in_recurrent_block
        * model.config.mean_recurrence
      )
      ln_f_index = (
        model.config.effective_expected_depth
      )

      for i in range(n_layers):
        # Prelude layers
        if i < model.config.n_layers_in_prelude:
          device_map[i] = model.hf_device_map["transformer.prelude"]

        # Core block layers (recurrent, may repeat device assignment)
        elif i < recurrent_block_end_index:
          device_map[i] = model.hf_device_map[
            f"transformer.core_block.{(i - model.config.n_layers_in_prelude) % model.config.n_layers_in_recurrent_block}"
          ]

        # Coda layers
        elif i < model.config.effective_expected_depth:
          device_map[i] = model.hf_device_map["transformer.coda"]

        elif i == ln_f_index:
          device_map[i] = model.hf_device_map["transformer.ln_f"]
        
        else:
          raise ValueError(f"Layer index {i} exceeds the expected number of layers {n_layers} for Huginn model.")

      return device_map
    case "llama":
      n_layers = get_n_layers(model)
      if len(model.hf_device_map) == 1:
        # If the model is on a single device, assign all layers to that device
        device = model.hf_device_map['']
        for i in range(n_layers):
          device_map[i] = device
        return device_map
      
      for i in range(n_layers):
        if i < model.config.num_hidden_layers:
          # Assign each layer to the device specified in the hf_device_map
          device_map[i] = model.hf_device_map[f"model.layers.{i}"]
        elif i == model.config.num_hidden_layers:
          device_map[i] = model.hf_device_map["model.norm"]
        else:
          raise ValueError(f"Layer index {i} exceeds the expected number of layers {n_layers} for Llama model.")
      return device_map
    case _:
      raise ValueError(f"Unsupported model type: {model.config.model_type}")