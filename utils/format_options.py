def format_options(option_texts: list[str]):
  """
  Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  output = "Options are:\n"
  option_labels = ["A", "B", "C", "D", "E", "F", "G", 
                   "H", "I", "J"]
  for label, text in zip(option_labels, option_texts):
    output += f"{label}: {text}\n"
  return output