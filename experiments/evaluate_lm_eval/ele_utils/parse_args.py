from typing import TypedDict
import argparse

class Args(TypedDict):
  use_local_datasets: bool
  data_path: str | None

  models_path: str
  model_name: str
  device: str
  with_parallelize: bool

  huginn_model_criterion: str | None
  huginn_num_steps: int | None

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
    '--use_local_datasets',
    action='store_true',
    help="Use local datasets instead of downloading from Hugging Face.",
  )
  parser.add_argument(
    '--data_path',
    type=str,
    help="Path to the local datasets directory. If specified, will use local datasets instead of downloading.",
    default=None,
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
    '--device',
    type=str,
    help="Device to run the model on, e.g., 'cuda', 'cpu', or 'auto' for automatic selection",
    default="auto",
  )
  parser.add_argument(
    '--with_parallelize',
    action='store_true',
    help="Whether to use parallel processing for evaluation. If set, will use lm_eval's parallelize function.",
    default=False,
  )

  parser.add_argument(
    '--huginn_model_criterion',
    type=str,
    help="Criterion for the model evaluation",
    default="entropy-diff",
  )
  parser.add_argument(
    '--huginn_num_steps',
    type=int,
    help="Number of steps for Huginn model evaluation. If None, will use the default behavior of the model.",
    default=32,
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
    '--output_file_path',
    type=str,
    help="Path to save the evaluation results",
    default=None,
  )

  args, _ = parser.parse_known_args()

  if not args["output_file_path"].endswith(".pt"):
    raise ValueError("Output file path must end with .pt")

  return args.__dict__