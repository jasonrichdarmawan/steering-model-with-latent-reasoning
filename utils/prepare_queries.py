from utils.format_options import format_options

from transformers import PreTrainedTokenizerBase

def prepare_queries(
  model_name: str,
  data,
  data_name: str,
  tokenizer: PreTrainedTokenizerBase | None = None,
  system_prompt: str | None = None,
  fewshot_prompts: dict[str, list[str]] | None = None,
  with_cot: bool = True,
):
  """
  model_name: The identifier for the model architecture, typically obtained from PreTrainedModel.config.model_type. 
    This determines how the query is formatted and which prompt templates or special handling are applied 
    (e.g., chat-style formatting for models starting with "huginn-").
  """
  queries: list[str] = []

  match data_name:
    case "mmlu-pro" | "mmlu-pro-3000samples":
      for entry in data:
        selected_fewshot_prompts = fewshot_prompts
        if selected_fewshot_prompts is not None:
          category = entry['category']
          selected_fewshot_prompts = fewshot_prompts[category]
        
        question = entry['question']
        options = format_options(entry['options'])
        question_content = f"{question}\n{options}"

        query = generate_query(
          model_name=model_name,
          question_content=question_content,
          tokenizer=tokenizer,
          system_prompt=system_prompt,
          fewshot_prompts=selected_fewshot_prompts,
          with_cot=with_cot
        )

        queries.append(query)
    case _:
      raise ValueError(f"Unsupported data name: {data_name}")

  return queries

def generate_query(
  model_name: str,
  question_content: str,
  tokenizer: PreTrainedTokenizerBase | None = None,
  system_prompt: str | None = None,
  fewshot_prompts: list[str] | None = None,
  with_cot: bool = True
):
  """
  Constructs the final query string for the model based on the provided parameters.

  Args:
    model_name: The identifier for the model architecture, typically obtained from PreTrainedModel.config.model_type. 
      This determines how the query is formatted and which prompt templates or special handling are applied 
      (e.g., chat-style formatting for models starting with "huginn-").
    question_content: The main content of the question to be asked (do NOT add "Q: " prefix; formatting is handled here).
    tokenizer: Optional tokenizer for formatting chat templates.
    system_prompt: Optional system prompt to guide the model's behavior.
    fewshot_prompts: Optional list of few-shot example prompts to include before the main question.
    with_cot: If True, encourages chain-of-thought reasoning by appending "Let's think step by step." to the prompt.

  Returns:
    The fully formatted query string ready for model inference.

  Notes:
    - For models starting with "huginn_", the function constructs a chat-style prompt using a system message and a user message.
    - If few-shot prompts are provided, they are prepended to the user message.
    - If chain-of-thought reasoning is enabled, the question is followed by "A: Let's think step by step." to encourage detailed reasoning.
    - Otherwise, the question is followed by "A: " for a direct answer.
    - The tokenizer's chat template is used to format the final query string.
  """
  query: str = ""

  match model_name:
    case name if name.startswith("huginn_"):
      """
      Reference: 
      [1] https://github.com/seal-rg/recurrent-pretraining/blob/0d9ed974d253e16498edec5c0c0916fdef4eb339/examples/inference_demo.ipynb
      [2] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
      """
      messages: list[str] = []
      
      if system_prompt == None:
        system_prompt = "You are a helpful assistant."
      
      messages.append({
        "role": "system",
        "content": system_prompt
      })

      user_content = ""

      if fewshot_prompts:
        for fewshot_prompt in fewshot_prompts:
          # Each few-shot prompt should already end with "\n\n"
          user_content += fewshot_prompt
      
      # If few-shot prompts are provided and chain-of-thought (CoT) reasoning is enabled (with_cot=True),
      # append the question followed by "A: Let's think step by step." to encourage step-by-step reasoning.
      # Otherwise, append the question followed by "A: " for a direct answer.
      if fewshot_prompts and with_cot:
        user_content += f"Q: {question_content}\nA: Let's think step by step. "
      else:
        user_content += f"Q: {question_content}\nA: "

      messages.append({
        "role": "user",
        "content": user_content
      })

      query: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
      )
    case _:
      raise ValueError(f"Unsupported model name: {model_name}")
    
  return query