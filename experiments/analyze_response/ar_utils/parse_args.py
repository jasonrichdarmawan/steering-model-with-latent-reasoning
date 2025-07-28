from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode

from typing import TypedDict
import argparse

class Args(TypedDict):
  model_path: str
  model_name: str

  huginn_num_steps: int | None

  batch_size: str

  use_linear_probes: bool
  linear_probes_file_path: str | None

  use_candidate_directions: bool
  candidate_directions_file_path: str | None

  layer_indices: list[int]
  direction_normalization_mode: DirectionNormalizationMode
  projection_hook_mode: ProjectionHookMode
  modification_mode: TokenModificationMode

  with_hidden_states_pre_hook: bool
  with_hidden_states_post_hook: bool

  with_attention_pre_hook: bool
  with_attention_post_hook: bool

  with_mlp_pre_hook: bool
  with_mlp_post_hook: bool

  scale: float

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Analyze model responses."
  )

  parser.add_argument(
    '--models_path',
    help="Path to the root directory containing multiple model folders",
    type=str,
  )
  parser.add_argument(
    '--model_name',
    help="Name of the model to analyze",
    type=str,
  )

  parser.add_argument(
    '--huginn_num_steps',
    help="Number of steps for Huginn model evaluation. If None, will use the default behavior of the model.",
    type=int,
    default=None,
  )

  parser.add_argument(
    '--batch_size',
    type=int,
    help="Batch size for processing the test dataset",
  )

  parser.add_argument(
    '--use_linear_probes',
    action="store_true",
    help="Whether to use linear probes for the intervention"
  )
  parser.add_argument(
    '--linear_probes_file_path',
    type=str,
    help="Path to the file containing the linear probes for intervention",
    default=None,
  )

  parser.add_argument(
    '--use_candidate_directions',
    action="store_true",
    help="Whether to use candidate directions for the intervention"
  )
  parser.add_argument(
    '--candidate_directions_file_path',
    type=str,
    help="Path to the file containing the candidate directions for intervention",
    default=None,
  )

  parser.add_argument(
    '--layer_indices',
    type=int,
    nargs='+',
    help="Indices of the layers to apply the intervention",
  )
  parser.add_argument(
    '--direction_normalization_mode',
    type=DirectionNormalizationMode,
    choices=list(DirectionNormalizationMode),
    help="Normalization mode for the candidate directions.",
  )
  parser.add_argument(
    '--projection_hook_mode',
    type=ProjectionHookMode,
    choices=list(ProjectionHookMode),
    help="Mode for the projection hook intervention.",
  )
  parser.add_argument(
    '--modification_mode',
    type=TokenModificationMode,
    choices=list(TokenModificationMode),
    help="Mode for modifying tokens during the intervention.",
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
    '--with_attention_pre_hook',
    action="store_true",
    help="Whether to use pre-hook for attention layers"
  )
  parser.add_argument(
    '--with_attention_post_hook',
    action="store_true",
    help="Whether to use post-hook for attention layers"
  )
  
  parser.add_argument(
    '--with_mlp_pre_hook',
    action="store_true",
    help="Whether to use pre-hook for MLP layers"
  )
  parser.add_argument(
    '--with_mlp_post_hook',
    action="store_true",
    help="Whether to use post-hook for MLP layers"
  )

  parser.add_argument(
    '--scale',
    type=float,
    help="Scale factor for the projection direction"
  )

  args = parser.parse_args()

  return args.__dict__