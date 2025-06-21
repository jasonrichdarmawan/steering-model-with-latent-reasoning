import re
import random
import warnings

def set_model_predict_correctness(
  entries: list[str],
  queries: list[str],
  responses: list[str],
  test_dataset_name: str
):
  match test_dataset_name:
    case 'mmlu-pro-3000samples.json':
      for entry, query, response in zip(entries, queries, responses):
        entry['solution'] = response
        
        prediction = get_prediction(
          output=response, 
          test_dataset_name=test_dataset_name
        )

        if prediction is None:
          warnings.warn(
            f"\n{'='*60}\n"
            f"Warning: No answer found.\n"
            f"Query:\n{query}\n"
            f"Answer: {entry['answer']}\n"
            f"Response:\n{response}\n"
            f"{'='*60}\n"
          )
          return random.choice(
            ['A', 'B', 'C', 'D', 'E', 
            'F', 'G', 'H', 'I', 'J']
          )

        if entry["answer"] == prediction:
          entry["model_predict_correctness"] = True
        else:
          entry["model_predict_correctness"] = False
    case _:
      raise ValueError(f"Unsupported test dataset name: {test_dataset_name}")

def get_prediction(
  output: str, 
  test_dataset_name: str
) -> str | None:
  match test_dataset_name:
    case 'mmlu-pro-3000samples.json':
      pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
      match = re.search(pattern, output)
      if match:
        return match.group(1)
      else:
        return None
    case _:
      raise ValueError(f"Unsupported test dataset name: {test_dataset_name}")