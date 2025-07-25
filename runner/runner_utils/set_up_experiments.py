from utils import CacheHiddenStatesMode
from utils import DirectionNormalizationMode
from utils import ProjectionHookMode
from utils import TokenModificationMode

from jobs import get_save_hidden_states
from jobs import get_evaluate_lm_eval
from jobs import get_mmlu
from jobs import get_mmlu_pro

def set_up_experiments(
  workspace_path: str,
  jobs: list[str]
) -> list[list[str]]:
  experiments = []

  for job in jobs:
    if job == "save_hidden_states":
      command = get_save_hidden_states(
        workspace_path=workspace_path,
        model_name="huginn-0125",
        cache_hidden_states_mode=CacheHiddenStatesMode.FIRST_ANSWER_TOKEN,
      )
      experiments.append([command])
    elif job == "evaluate_lm_eval_tasks_piqa":
      command = get_evaluate_lm_eval(
        workspace_path=workspace_path,
        model_name="huginn-0125",
        tasks="piqa",
        num_fewshot=0,
        with_intervention=False,
      )
      experiments.append([command])
    elif job == "evaluate_lm_eval_tasks_piqa_use_linear_probes":
      command = get_evaluate_lm_eval(
        workspace_path=workspace_path,
        model_name="huginn-0125",
        tasks="piqa",
        num_fewshot=0,

        with_intervention=True,
        
        use_linear_probes=True,

        layer_indices=[31],
        direction_normalization_mode=DirectionNormalizationMode.UNIT_VECTOR,
        projection_hook_mode=ProjectionHookMode.FEATURE_ADDITION,
        modification_mode=TokenModificationMode.LAST_TOKEN,
        with_hidden_states_pre_hook=False,
        with_hidden_states_post_hook=True,
        scale=1.0,
      )
      experiments.append([command])
    elif job.startswith("mmlu_pro"):
      commands = get_mmlu_pro(workspace_path=workspace_path, job=job,)
      experiments.append(commands)
    elif job.startswith("mmlu"):
      commands = get_mmlu(workspace_path=workspace_path, job=job,)
      experiments.append(commands)
    else:
      raise ValueError(f"Unknown job: {job}")

  return experiments