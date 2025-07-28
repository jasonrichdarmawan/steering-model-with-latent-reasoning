from utils.format_options import format_options

from transformers import PreTrainedTokenizerBase

def prepare_queries(
  model_name: str,
  data,
  data_name: str,
  tokenizer: PreTrainedTokenizerBase | None = None,
  apply_question_format: bool = True,
  apply_chat_template: bool = False,
  system_prompt: str | None = None,
  fewshot_prompts: dict[str, list[str]] | None = None,
  with_cot: bool = True,
  with_options: bool = True,
):
  """
  model_name: The identifier for the model architecture, typically obtained from PreTrainedModel.config.model_type. 
    This determines how the query is formatted and which prompt templates or special handling are applied 
    (e.g., chat-style formatting for models starting with "huginn-").
  """
  queries: list[str] = []

  match data_name:
    case "mmlu-pro-3000samples.json":
      for entry in data:
        # Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/LiReFs_storing_hs.ipynb
        question = entry['question']
        question_content = question
        if with_options:
          options = format_options(entry['options'])
          question_content = f"{question}\n{options}"

        query = prepare_query(
          model_name=model_name,
          question_content=question_content,
          tokenizer=tokenizer,
          apply_question_format=apply_question_format,
          apply_chat_template=apply_chat_template,
          system_prompt=system_prompt,
          fewshot_prompts=fewshot_prompts[entry['category']] if fewshot_prompts else None,
          with_cot=with_cot,
        )

        queries.append(query)
      
      return queries
    case _:
      raise ValueError(f"Unsupported data name: {data_name}")

def prepare_query(
  model_name: str,
  question_content: str,
  tokenizer: PreTrainedTokenizerBase | None = None,
  apply_question_format: bool = True,
  apply_chat_template: bool = False,
  system_prompt: str | None = None,
  fewshot_prompts: list[str] | None = None,
  with_cot: bool = True
):
  """
  Constructs the final query string for the model based on the provided parameters.

  Args:
    model_name: The identifier for the model architecture, typically obtained from PreTrainedModel.config.model_type. 
    question_content: The main content of the question to be asked (do NOT add "Q: " prefix; formatting is handled here).
    tokenizer: Optional tokenizer for formatting chat templates.
    apply_chat_template: If True, applies a chat template to the query, which is useful for models that expect chat-style input.
    system_prompt: Optional system prompt to guide the model's behavior.
    fewshot_prompts: Optional list of few-shot example prompts to include before the main question.
    with_cot: If True, encourages chain-of-thought reasoning by appending "Let's think step by step." to the prompt.

  Returns:
    The fully formatted query string ready for model inference.
  """
  match model_name:
    # TODO: assess whether we need
    # to handle each model type separately
    case "huginn_raven":
      query = _prepare_query_huginn(
        question_content=question_content,
        tokenizer=tokenizer,
        apply_question_format=apply_question_format,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
        fewshot_prompts=fewshot_prompts,
        with_cot=with_cot
      )
      
      return query
    case "llama":
      query = _prepare_query_llama(
        question_content=question_content,
        tokenizer=tokenizer,
        apply_question_format=apply_question_format,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
        fewshot_prompts=fewshot_prompts,
        with_cot=with_cot,
      )

      return query
    case _:
      raise ValueError(f"Unsupported model name: {model_name}")

def _prepare_query_llama(
  question_content: str,
  tokenizer: PreTrainedTokenizerBase | None = None,
  apply_question_format: bool = True,
  apply_chat_template: bool = False,
  system_prompt: str | None = None,
  fewshot_prompts: list[str] | None = None,
  with_cot: bool = True
):
  """
  Reference: https://github.com/yihuaihong/Linear_Reasoning_Features/blob/4f95e7e82935b1bce05e5cda4fc0ca8eff648d98/reasoning_representation/LiReFs_storing_hs.ipynb
  """
  user_content = ""

  if fewshot_prompts:
    for fewshot_prompt in fewshot_prompts:
      user_content += f"{fewshot_prompt}\n\n"
  
  if apply_question_format:
    if fewshot_prompts and with_cot:
      user_content += f"Q: {question_content}\nA: Let's think step by step. "
    else:
      user_content += f"Q: {question_content}\nA: "
  else:
    user_content += question_content

  if apply_chat_template is False:
    query = user_content
    return query

  if apply_chat_template:
    raise NotImplementedError("Chat template formatting is not implemented for Llama models. ")
  
  if system_prompt:
    raise NotImplementedError("System prompts are not supported for Llama models. ")

def _prepare_query_huginn(
  question_content: str,
  tokenizer: PreTrainedTokenizerBase | None = None,
  apply_question_format: bool = True,
  apply_chat_template: bool = False,
  system_prompt: str | None = None,
  fewshot_prompts: list[str] | None = None,
  with_cot: bool = True
):
  """
  Reference: 
  [1] https://github.com/seal-rg/recurrent-pretraining/blob/0d9ed974d253e16498edec5c0c0916fdef4eb339/examples/inference_demo.ipynb
  [2] https://github.com/yihuaihong/Linear_Reasoning_Features/blob/f23f2547862a2c1b57f1cfa3c547776cb38f762a/reasoning_representation/Intervention/utils.py
  """
  user_content = ""

  if fewshot_prompts:
    for fewshot_prompt in fewshot_prompts:
      user_content += f"{fewshot_prompt}\n\n"
  
  if apply_question_format:
    # If few-shot prompts are provided and chain-of-thought (CoT) reasoning is enabled (with_cot=True),
    # append the question followed by "A: Let's think step by step." to encourage step-by-step reasoning.
    # Otherwise, append the question followed by "A: " for a direct answer.
    if fewshot_prompts and with_cot:
      user_content += f"Q: {question_content}\n\nA: Let's think step by step. "
    else:
      user_content += f"Q: {question_content}\n\nA: "
  else:
    user_content += question_content

  if apply_chat_template is False:
    query = user_content
    return query

  messages: list[dict[str, str]] = []
  
  if system_prompt:
    messages.append({
      "role": "system",
      "content": system_prompt
    })

  messages.append({
    "role": "user",
    "content": user_content
  })

  query: str = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )

  return query