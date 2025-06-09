import torch
from tqdm import tqdm

def cache_hidden_states(data,
                        tokenizer,
                        model,
                        data_batch_size: int,):
  n_layers = get_n_layers(model)
  n_embd = get_n_embd(model)

  n_layers_to_cache = list(range(n_layers))

  hidden_states_cache = {
    index: torch.empty((0, n_embd))
    for index in n_layers_to_cache
  }

  data_sample_size = len(data)

  queries = []
  for index, entry in tqdm(enumerate(data)):
    query = "Q: " + entry['question'] + "\nA: "
    queries.append(query)

    if (
      len(queries) == data_batch_size
      or index == data_sample_size - 1
    ):
      match model.config.model_type:
        case name if name.startswith("huginn_"):
          inputs = tokenizer(
            queries, return_tensors="pt",
            padding="longest", 
            return_token_type_ids=False
          ).input_ids.to("cuda")

          with (
            torch.no_grad(), 
            torch.autocast(
              device_type="cuda", dtype=torch.bfloat16
            )
          ):
            output_details = {
              "return_logits": False,
              "return_latents": False,
              "return_attention": False,
              "return_head": True,
              "return_stats": False,
            }
            outputs = model(inputs, output_details=output_details)

      for layer in n_layers_to_cache:
        hidden_states_cache[layer] = torch.cat(
          (
            hidden_states_cache[layer],
            outputs["hidden_states"][layer][:, -1, :].cpu()
          ),
          dim=0
        )

      queries = []
    torch.cuda.empty_cache()
    
  return hidden_states_cache

def get_n_layers(model) -> int:
  match model.config.model_type:
    case name if name.startswith("huginn_"):
      return model.config.effective_expected_depth
    case _:
      return 0

def get_n_embd(model) -> int:
  match model.config.model_type:
    case name if name.startswith("huginn_"):
      return model.config.n_embd
    case _:
      return 0