from jobs import handle_save_hidden_states
from jobs import handle_save_candidate_directions
from jobs import handle_analyze_steering_effect_per_layer
from jobs import handle_train_linear_probes
from jobs import handle_evaluate_accuracy_reasoning_memorizing
from jobs import handle_evaluate_lm_eval
from jobs import get_mmlu
from jobs import get_mmlu_pro

def set_up_experiments(
  workspace_path: str,
  jobs: list[str]
) -> list[list[str]]:
  experiments = []

  for job in jobs:
    if job.startswith("save_hidden_states"):
      command = handle_save_hidden_states(
        workspace_path=workspace_path,
        job=job,
      )
      experiments.append([command])
    elif job.startswith("save_candidate_directions"):
      command = handle_save_candidate_directions(
        workspace_path=workspace_path,
        job=job,
      )
      experiments.append([command])
    elif job.startswith("analyze_steering_effect_per_layer"):
      command = handle_analyze_steering_effect_per_layer(
        workspace_path=workspace_path,
        job=job,
      )
      experiments.append([command])
    elif job.startswith("train_linear_probes"):
      command = handle_train_linear_probes(
        workspace_path=workspace_path,
        job=job,
      )
      experiments.append([command])
    elif job.startswith("evaluate_accuracy_reasoning_memorizing"):
      command = handle_evaluate_accuracy_reasoning_memorizing(
        workspace_path=workspace_path,
        job=job,
      )
      experiments.append([command])
    elif job.startswith("evaluate_lm_eval"):
      command = handle_evaluate_lm_eval(
        workspace_path=workspace_path,
        job=job,
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