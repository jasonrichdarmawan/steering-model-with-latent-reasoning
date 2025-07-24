from typing import TypedDict
import argparse

class Args(TypedDict):
  device: str

  hidden_states_file_path: str

  epochs: int

  lr: float
  weight_decay: float

  sample_size: int | None
  test_ratio: float
  batch_size: int

  moving_average_window_size_step: int | None
  moving_average_window_size_epoch: int | None

  output_dir: str | None
  checkpoint_freq: int | None
  use_early_stopping: bool
  early_stopping_patience: int | None

def parse_args() -> Args:
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--device',
    type=str,
    help="Device to run the training on, e.g., 'cuda:0' or 'cpu'.",
  )

  parser.add_argument(
    '--hidden_states_file_path',
    type=str,
    help="File path to save the hidden states.",
  )

  parser.add_argument(
    '--epochs',
    type=int,
    help="Number of epochs to train the linear probe.",
  )

  parser.add_argument(
    '--lr',
    type=float,
    help="Learning rate for training the linear probe.",
  )
  parser.add_argument(
    '--weight_decay',
    type=float,
    help="Weight decay for the optimizer.",
  )

  parser.add_argument(
    '--sample_size',
    type=int,
    help="Number of samples to use for training the linear probe. If not specified, all samples will be used.",
    default=None,
  )
  parser.add_argument(
    '--test_ratio',
    type=float,
    help="Proportion of the dataset to include in the test split. Default is 0.25.",
    default=0.25,
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    help="Batch size for training the linear probe.",
  )

  parser.add_argument(
    '--moving_average_window_size_step',
    type=int,
    help="Window size for moving average of losses during training. If not specified, no moving average will be applied.",
    default=None,
  )
  parser.add_argument(
    '--moving_average_window_size_epoch',
    type=int,
    help="Window size for moving average of losses per epoch. If not specified, no moving average will be applied.",
    default=None,
  )

  parser.add_argument(
    '--output_dir',
    type=str,
    help="Directory to save the output files, including checkpoints.",
    default=None,
  )
  parser.add_argument(
    '--checkpoint_freq',
    type=int,
    help="Frequency of saving checkpoints during training.",
    default=None,
  )
  parser.add_argument(
    '--use_early_stopping',
    action='store_true',
    help="Whether to use early stopping during training.",
  )
  parser.add_argument(
    '--early_stopping_patience',
    type=int,
    help="Number of epochs with no improvement after which training will be stopped. Default is 10.",
    default=None,
  )

  args = parser.parse_args()
  
  return args.__dict__