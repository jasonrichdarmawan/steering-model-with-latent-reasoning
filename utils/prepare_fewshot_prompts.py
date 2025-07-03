import os
from datasets import load_from_disk

from utils.format_options import format_options

def prepare_fewshot_prompts(
  data_path: str,
  data_name: str,
  with_cot: bool = True,
):
  """
  Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """

  match data_name:
    case 'mmlu-pro':
      dataset = load_from_disk(
        dataset_path=os.path.join(data_path, data_name),
      )
      val_dataset = dataset['validation']
      
      fewshot_prompts: dict[
        str, list[str]
      ] = {
        category: []
        for category in set(val_dataset['category'])
      }

      for entry in val_dataset:
        category = entry["category"]
        question = entry["question"]
        options = format_options(entry["options"])
        
        if with_cot:
          # Format the few-shot prompt examples:
          # - example_cot includes the question, options, and the chain-of-thought (CoT) reasoning content.
          #   The CoT content in entry['cot_content'] already starts with "A: Let's think step by step. ...".
          # - example_no_cot includes the question, options, and a direct answer in the format "A: The answer is (answer)."
          fewshot_prompt = f"Q: {question}\n{options}\n{entry['cot_content']}"
        else:
          # Note: The original code does not include the prefix "A: " before the answer.
          # Here, we do not follow the original format and only prepend "The answer is (...)." after the options.
          fewshot_prompt = f"Q: {question}\n{options}\nA: The answer is ({entry['answer']})."
        
        fewshot_prompts[category].append(fewshot_prompt)
      
      return fewshot_prompts
    case _:
      return None