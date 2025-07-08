from utils import ProcessHiddenStatesMode
from utils import DirectionNormalizationMode
from utils import ProjectionHookMode

from typing import TypedDict
import argparse

class Args(TypedDict):
  data_path: str | None

  models_path: str
  model_name: str

  huginn_mean_recurrence: int | None

  with_intervention: bool
  process_hidden_states_mode: ProcessHiddenStatesMode | None
  candidate_directions_file_path: str | None
  direction_normalization_mode: DirectionNormalizationMode | None
  projection_hook_mode: ProjectionHookMode | None
  layer_indices: list[int] | None
  with_hidden_states_pre_hook: bool
  with_hidden_states_post_hook: bool
  scale: float | None

  tasks: str
  num_fewshot: int
  batch_size: int | str # 'auto' or int
  limit: int
  output_file_path: str | None

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Evaluate language model performance using lm-eval."
  )

  parser.add_argument(
    '--data_path',
    type=str,
    help="Path to the local datasets directory",
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
  )

  parser.add_argument(
    '--huginn_mean_recurrence',
    type=int,
    help="Number of mean recurrence for Huginn model evaluation",
    default=None,
  )

  parser.add_argument(
    '--with_intervention',
    action='store_true',
    help="Whether to apply intervention during evaluation.",
  )
  parser.add_argument(
    '--process_hidden_states_mode',
    type=ProcessHiddenStatesMode,
    choices=list(ProcessHiddenStatesMode),
    help="Mode for processing hidden states during the intervention.",
    default=None,
  )
  parser.add_argument(
    '--candidate_directions_file_path',
    type=str,
    help="Path to the candidate directions file. If specified, will load the candidate directions from this file.",
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
    '--layer_indices',
    type=int,
    nargs='+',
    help="Indices of the layers to apply the intervention",
    default=None,
  )
  parser.add_argument(
    '--with_hidden_states_pre_hook',
    action='store_true',
    help="Whether to apply the hidden states pre-hook intervention",
  )
  parser.add_argument(
    '--with_hidden_states_post_hook',
    action='store_true',
    help="Whether to apply the hidden states post-hook intervention",
  )
  parser.add_argument(
    '--scale',
    type=float,
    help="Scale factor for the intervention. If None, will use the default scale of 0.1.",
    default=None,
  )

  parser.add_argument(
    '--tasks',
    type=str,
    help="Comma-separated list of tasks to evaluate the model on",
  )
  parser.add_argument(
    '--num_fewshot',
    type=int,
    help="Number of few-shot examples to use for evaluation",
  )
  parser.add_argument(
    '--batch_size',
    type=lambda x: int(x) if x.isdigit() else x,
    help='Batch size for evaluation (int or "auto")',
  )
  parser.add_argument(
    '--limit',
    type=int,
    help="Limit the number of samples to evaluate",
    default=None,
  )
  parser.add_argument(
    '--output_file_path',
    type=str,
    help="Path to save the evaluation results",
    default=None,
  )

  args = parser.parse_args()

  return args.__dict__