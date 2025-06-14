from datasets import load_from_disk
from os.path import join
import random

def load_and_sample_test_dataset(
  data_path: str, 
  data_name: str,
  sample_size: int | None = None
):
  """
  Load the test dataset from a specified path and sample a given number of entries.
  Args:
    data_path (str): Path to the root directory containing multiple data folders.
    data_name (str): Folder name of the specific dataset to load from the root directory.
    sample_size (int, optional): Number of samples to randomly select from the test dataset. If None, sampling will be ignored and the full dataset will be returned.
  """
  dataset = load_from_disk(
    dataset_path=join(data_path, data_name),
  )
  dataset = list(dataset['test'])

  if sample_size is not None:
    return random.sample(dataset, sample_size)
  return dataset
