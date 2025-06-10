import json
from datasets import load_from_disk
from os.path import join
import random

def load_and_sample_test_dataset(data_file_path: str,
                                 data_path: str, 
                                 data_name: str,
                                 sample_size: int | None):
  """
  Load the test dataset from a specified path and sample a given number of entries.
  Args:
      data_file_path (str): Path to the JSON file containing the dataset. If not empty, this will be used and data_path and data_name will be ignored.
      data_path (str): Path to the root directory containing multiple data folders. Ignored if data_file_path is not empty.
      data_name (str): Folder name of the specific dataset to load from the root directory. Ignored if data_file_path is not empty.
      sample_size (int, optional): Number of samples to randomly select from the test dataset. If None, sampling will be ignored and the full dataset will be returned.
  """
  if data_file_path:
    with open(data_file_path, 'r', encoding='utf-8') as f:
      dataset = json.load(f)
    if sample_size is None:
      return dataset
    return random.sample(dataset, sample_size)

  dataset = load_from_disk(
    dataset_path=join(data_path, data_name),
  )

  test_dataset = list(dataset['test'])
  
  sampled_data = random.sample(test_dataset, sample_size)
  
  return sampled_data