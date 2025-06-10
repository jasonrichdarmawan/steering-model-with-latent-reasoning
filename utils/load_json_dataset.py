import json

def load_json_dataset(file_path: str):
  with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data