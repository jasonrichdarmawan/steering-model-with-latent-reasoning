from typing import TypedDict
import argparse

class Args(TypedDict):
  models_path: str
  model_name: str
  device: str

  huginn_num_steps: int | None
  
  hidden_states_cache_file_path: str
  mmlu_pro_3000samples_data_file_path: str

  test_data_path: str
  test_data_name: str
  with_fewshot_prompts: bool
  with_cot: bool
  batch_size: int

  with_intervention: bool
  layer_indices: list[int]
  with_pre_hook: bool
  with_post_hook: bool
  scale: float

  output_file_path: str | None

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Evaluate reasoning and memorization accuracy."
  )

  parser.add_argument(
    '--models_path',
    type=str,
    help="Path to the root directory containing multiple model folders",
  )
  parser.add_argument(
    '--model_name', 
    type=str,
    help="Folder name of the specific model to load from the root directory",
    default="huginn-0125"
  )
  parser.add_argument(
    '--device', 
    type=str, 
    default="auto",
    help="Device to run the model on, e.g., 'cuda', 'cpu', or 'auto' for automatic selection"
  )
  
  parser.add_argument(
    '--huginn_num_steps', 
    type=int, 
    default=None,
    help="Number of steps for Huginn model evaluation. If None, will use the default behavior of the model."
  )

  parser.add_argument(
    '--hidden_states_cache_file_path',
    type=str,
    help="Path to the cached hidden states",
  )
  parser.add_argument(
    '--mmlu_pro_3000samples_data_file_path', 
    type=str,
    help="Path to the MMLU Pro 3000 samples dataset file",
  )

  parser.add_argument(
    '--test_data_path', 
    type=str,
    help="Path to the root directory containing multiple data folders",
  )
  parser.add_argument(
    '--test_data_name', type=str,
    help="Folder name of the specific dataset to load from the root directory",
    default="mmlu-pro-3000samples"
  )
  parser.add_argument(
    '--with_fewshot_prompts',
    action="store_true",
    help="Whether to use few-shot prompts in the queries"
  )
  parser.add_argument(
    '--with_cot',
    action="store_true",
    help="Whether to use chain-of-thought (CoT) reasoning in the prompts"
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help="Batch size for processing the test dataset"
  )

  parser.add_argument(
    '--with_intervention',
    action="store_true",
    help="Whether to apply the intervention for reasoning and memorization accuracy evaluation"
  )
  parser.add_argument(
    '--layer_indices',
    type=int,
    nargs='+',
    help="Indices of the layers to apply the intervention",
    default=[66]
  )
  parser.add_argument(
    '--with_pre_hook',
    action="store_true",
    help="Whether to use pre-hook for the intervention"
  )
  parser.add_argument(
    '--with_post_hook',
    action="store_true",
    help="Whether to use post-hook for the intervention"
  )
  parser.add_argument(
    '--scale',
    type=float,
    default=0.1,
    help="Scale factor for the projection direction"
  )
  
  parser.add_argument(
    '--output_file_path',
    type=str,
    default=None,
    help="Path to save the output results. If None, results will not be saved to a file."
  )

  args, _ = parser.parse_known_args()
  
  return args.__dict__