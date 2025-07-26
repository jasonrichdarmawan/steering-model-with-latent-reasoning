from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode

from typing import TypedDict
import argparse

class Args(TypedDict):
  models_path: str
  model_name: str

  device: str | None

  huginn_num_steps: int | None

  test_data_path: str
  test_data_name: str
  test_data_sample_size: int | None
  with_fewshot_prompts: bool
  with_cot: bool
  batch_size: int

  with_intervention: bool

  candidate_directions_file_path: str | None

  layer_indices: list[int] | None
  direction_normalization_mode: DirectionNormalizationMode | None
  projection_hook_mode: ProjectionHookMode | None
  modification_mode: TokenModificationMode | None
  with_hidden_states_pre_hook: bool
  with_hidden_states_post_hook: bool
  scale: float | None

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
  )

  parser.add_argument(
    '--device',
    help="Device to run the evaluation on (e.g., 'cuda:0', 'cpu')",
    type=str,
    default=None,
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
  )
  parser.add_argument(
    '--test_data_sample_size',
    type=int,
    help="Number of samples to randomly select from the test dataset",
    default=None,
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
    help="Batch size for processing the test dataset",
  )

  parser.add_argument(
    '--with_intervention',
    action="store_true",
    help="Whether to apply the intervention for reasoning and memorization accuracy evaluation"
  )
  parser.add_argument(
    '--candidate_directions_file_path',
    type=str,
    help="Path to the file containing the candidate directions for intervention",
    default=None,
  )
  parser.add_argument(
    '--direction_normalization_mode',
    type=DirectionNormalizationMode,
    choices=list(DirectionNormalizationMode),
    help="Normalization mode for the candidate directions.",
    default=None, 
  )
  parser.add_argument(
    '--projection_hook_mode',
    type=ProjectionHookMode,
    choices=list(ProjectionHookMode),
    help="Mode for the projection hook intervention.",
    default=None,
  )
  parser.add_argument(
    '--modification_mode',
    type=TokenModificationMode,
    choices=list(TokenModificationMode),
    help="Mode for modifying tokens during the intervention.",
    default=None,
  )
  parser.add_argument(
    '--layer_indices',
    type=int,
    nargs='+',
    help="Indices of the layers to apply the intervention",
    default=None
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
    default=None,
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