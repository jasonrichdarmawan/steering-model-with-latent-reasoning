from typing import TypedDict
import argparse

class Args(TypedDict):
  models_path: str
  model_name: str

  huginn_num_steps: int | None

  test_data_path: str
  test_data_name: str
  test_data_sample_size: int | None
  with_fewshot_prompts: bool
  with_cot: bool
  batch_size: int

  with_intervention: bool
  hidden_states_data_file_path: str | None
  hidden_states_cache_file_path: str | None
  layer_indices: list[int]
  with_hidden_states_pre_hook: bool
  with_hidden_states_post_hook: bool
  scale: float

  output_file_path: str | None

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Evaluate reasoning and memorization accuracy."
  )

  parser.add_argument(
    '--models_path',
    help="Path to the root directory containing multiple model folders",
    type=str,
  )
  parser.add_argument(
    '--model_name', 
    help="Folder name of the specific model to load from the root directory",
    type=str,
    default="huginn-0125",
  )
  
  parser.add_argument(
    '--huginn_num_steps', 
    help="Number of steps for Huginn model evaluation. If None, will use the default behavior of the model.",
    type=int, 
    default=None,
  )

  parser.add_argument(
    '--test_data_path', 
    type=str,
    help="Path to the root directory containing multiple data folders",
  )
  parser.add_argument(
    '--test_data_name',
    help="Folder name of the specific dataset to load from the root directory",
    type=str,
    default="mmlu-pro-3000samples.json",
  )
  parser.add_argument(
    '--test_data_sample_size',
    type=int,
    help="Number of samples to randomly select from the test dataset",
    default=200,
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
    '--hidden_states_data_file_path',
    type=str,
    help="Path to the JSON file containing the hidden states dataset for intervention",
    default=None,
  )
  parser.add_argument(
    '--hidden_states_cache_file_path',
    type=str,
    help="Path to the cached hidden states",
    default=None,
  )
  parser.add_argument(
    '--layer_indices',
    type=int,
    nargs='+',
    help="Indices of the layers to apply the intervention",
    default=[66]
  )
  parser.add_argument(
    '--with_hidden_states_pre_hook',
    action="store_true",
    help="Whether to use pre-hook for the intervention"
  )
  parser.add_argument(
    '--with_hidden_states_post_hook',
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