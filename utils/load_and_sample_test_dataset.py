from datasets import load_from_disk
from os.path import join
import random

def load_and_sample_test_dataset(data_path: str, 
                                 data_name: str,
                                 sample_size: int = 600):
  dataset = load_from_disk(
    dataset_path=join(data_path, data_name),
  )

  test_dataset = list(dataset['test'])
  
  sampled_data = random.sample(test_dataset, sample_size)
  
  return sampled_data