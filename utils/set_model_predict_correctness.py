import re
from typing import NamedTuple

class NoAnswerWarning(NamedTuple):
  query: str
  response: str
  expected_answer: str

  def __str__(self):
    return (
      f"\n{'='*20}\n"
      f"Warning: No answer found.\n"
      f"Query:\n{self.query}\n"
      f"\n{' '*20}{'='*20}\n"
      f"Expected Answer:\n{self.expected_answer}\n"
      f"\n{' '*40}{'='*20}\n"
      f"Response:\n{self.response}\n"
      f"{'='*60}\n"
    )

def set_model_predict_correctness(
  entries: list[str],
  queries: list[str],
  responses: list[str],
  test_dataset_name: str,
  no_answers: list[NoAnswerWarning] | None = None,
):
  if no_answers is None:
    no_answers: list[NoAnswerWarning] = []

  match test_dataset_name:
    case 'mmlu-pro-3000samples.json':
      for entry, query, response in zip(entries, queries, responses):
        entry['solution'] = response
        
        prediction = get_prediction(
          output=response, 
          test_dataset_name=test_dataset_name
        )

        if prediction is None:
          no_answer = NoAnswerWarning(
            query=query,
            response=response,
            expected_answer=entry['answer']
          )
          no_answers.append(no_answer)

        if entry["answer"] == prediction:
          entry["model_predict_correctness"] = True
        else:
          entry["model_predict_correctness"] = False

      return no_answers
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