from typing import TypedDict
import argparse

class Args(TypedDict):
  use_hf_mirror: bool

  models_path: str
  model_name: str

  huginn_model_criterion: str | None

  tasks: str
  num_fewshot: int
  batch_size: int | str # 'auto' or int
  limit: int
  huginn_num_steps: int | None
  output_path: str | None

def parse_args() -> Args:
  parser = argparse.ArgumentParser(
    description="Evaluate language model performance using lm-eval."
  )

  parser.add_argument(
    '--use_hf_mirror',
    action='store_true',
    help="Use Hugging Face mirror for downloading models and datasets",
  )

  parser.add_argument(
    '--models_path',
    type=str,
    help="Path to the root directory containing multiple model folders",
    default="/home/npu-tao/jason/transformers",
  )
  parser.add_argument(
    '--model_name',
    type=str,
    help="Folder name of the specific model to load from the root directory",
    default="huginn-0125",
  )
  parser.add_argument(
    '--huginn_model_criterion',
    type=str,
    help="Criterion for the model evaluation",
    default="entropy-diff",
  )

  parser.add_argument(
    '--tasks',
    type=str,
    help="Comma-separated list of tasks to evaluate the model on",
    default="mmlu",
  )
  parser.add_argument(
    '--num_fewshot',
    type=int,
    help="Number of few-shot examples to use for evaluation",
    default=5,
  )
  parser.add_argument(
    '--batch_size',
    type=lambda x: int(x) if x.isdigit() else x,
    help='Batch size for evaluation (int or "auto")',
    default=4,
  )
  parser.add_argument(
    '--limit',
    type=int,
    help="Limit the number of samples to evaluate",
    default=50,
  )
  parser.add_argument(
    '--huginn_num_steps',
    type=int,
    help="Number of recurrence steps for the recurrent model (e.g., huginn-0125). Each core block will be iterated this many times; 32 means 32 iterations per block.",
    default=None
  )
  parser.add_argument(
    '--output_path',
    type=str,
    help="Path to save the evaluation results",
    default=None,
  )

  args, _ = parser.parse_known_args()

  return args.__dict__