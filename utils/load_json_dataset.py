import json
import random

def load_json_dataset(
  file_path: str,
  sample_size: int | None = None
):
  with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  if sample_size is not None:
    data = random.sample(data, sample_size)
  return data