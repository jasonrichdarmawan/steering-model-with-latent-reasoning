from typing import TypedDict
import argparse

class Args(TypedDict):
  models_path: str
  model_name: str
  
  hidden_states_cache_path: str

  test_data_path: str
  test_data_name: str
  with_fewshot_prompts: bool
  with_cot: bool
  batch_size: int

  with_intervention: bool
  layer_indices: list[int]
  with_pre_hook: bool
  with_post_hook: bool
  scale: float

def parse_args() -> Args:
  parser = argparse.ArgumentParser(description="Evaluate reasoning and memorization accuracy.")

  parser.add_argument('--models_path', type=str,
                      help="Path to the root directory containing multiple model folders",
                      default="/root/autodl-fs/transformers")
  parser.add_argument('--model_name', type=str,
                      help="Folder name of the specific model to load from the root directory",
                      default="huginn-0125")
  
  parser.add_argument('--hidden_states_cache_path', type=str,
                      help="Path to the cached hidden states",
                      default="/root/autodl-fs/experiments/hidden_states_cache")
  
  parser.add_argument('--test_data_path', type=str,
                      help="Path to the root directory containing multiple data folders",
                      default="/root/autodl-fs/datasets")
  parser.add_argument('--test_data_name', type=str,
                      help="Folder name of the specific dataset to load from the root directory",
                      default="mmlu-pro-3000samples")
  parser.add_argument('--with_fewshot_prompts', action="store_true",
                      help="Whether to use few-shot prompts in the queries")
  parser.add_argument('--with_cot', action="store_true",
                      help="Whether to use chain-of-thought (CoT) reasoning in the prompts")
  parser.add_argument('--batch_size', type=int, default=1,
                      help="Batch size for processing the test dataset")

  parser.add_argument('--with-intervention', action="store_true",
                      help="Whether to apply the intervention for reasoning and memorization accuracy evaluation")
  parser.add_argument('--layer_indices', type=int, nargs='+',
                      help="Indices of the layers to apply the intervention",
                      default=[66])
  parser.add_argument('--with_pre_hook', action="store_true",
                      help="Whether to use pre-hook for the intervention")
  parser.add_argument('--with_post_hook', action="store_true",
                      help="Whether to use post-hook for the intervention")
  parser.add_argument('--scale', type=float, default=0.1,
                      help="Scale factor for the projection direction")

  args, _ = parser.parse_known_args()
  
  return {
    "models_path": args.models_path,
    "model_name": args.model_name,
    
    "hidden_states_cache_path": args.hidden_states_cache_path,
    
    "test_data_path": args.test_data_path,
    "test_data_name": args.test_data_name,
    "with_fewshot_prompts": args.with_fewshot_prompts,
    "with_cot": args.with_cot,
    "batch_size": args.batch_size,
    
    "with_intervention": args.with_intervention,
    "layer_indices": args.layer_indices,
    "with_pre_hook": args.with_pre_hook,
    "with_post_hook": args.with_post_hook,
    "scale": args.scale,
  }