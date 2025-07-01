from jaxtyping import Float
import torch
from torch import Tensor
from matplotlib import pyplot as plt

def compute_create_save_cosine_similarities_plot(
  candidate_directions: dict[int, Float[Tensor, "n_embd"]],
  x2: Float[Tensor, "n_embd"],
  x2_name: str,
  output_file_path: str | None = None,
):
  """
  High cosine similarity between candidate directions 
  and model embedding or unembedding layers can indicate 
  that the model retains token representation information
  in early layers rather than behavioral patterns. 
  In other words, the effect value computed in these layers
  can be misleading.

  Reference: https://github.com/cvenhoff/steering-thinking-llms/blob/83fc94a7c0afd9d96ca418897d8734a8b94ce848/train-steering-vectors/cosine_sim.py
  """
  cosine_similarities = compute_cosine_similarities(
    candidate_directions=candidate_directions,
    x2=x2,
  )

  print(f"Layers by the highest cosine similarity between candidate directions and {x2_name}:")
  top_cosine_similarities = sorted(
    cosine_similarities.items(),
    key=lambda item: item[1],
    reverse=True,
  )
  for layer_index, cosine_similarity in top_cosine_similarities:
    print(f"Layer {layer_index}: {cosine_similarity:.4f}")

  cosine_similarities_fig = create_cosine_similarity_plot(
    cosine_similarities=cosine_similarities,
    x2_name=x2_name,
  )

  if output_file_path is None:
    print("No output file path specified. Plot will not be saved.")
    return
  
  print(f"Saving the cosine similarity between candidate directions and {x2_name} plot to: {output_file_path}")
  cosine_similarities_fig.savefig(
    fname=output_file_path,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.1,
  )

def compute_cosine_similarities(
  candidate_directions: dict[int, Float[Tensor, "n_embd"]],
  x2: Float[Tensor, "n_embd"],
):
  cosine_similarities: dict[int, float] = {}
  for layer_index, candidate_direction in candidate_directions.items():
    cosine_similarity = torch.cosine_similarity(
      x1=candidate_direction.to(device=x2.device),
      x2=x2,
      dim=-1
    ).max().item()
    cosine_similarities[layer_index] = cosine_similarity
  return cosine_similarities

def create_cosine_similarity_plot(
  cosine_similarities: dict[int, float],
  x2_name: str,
):
  fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(12, 6),
  )

  ax.plot(
    range(len(cosine_similarities)),
    list(cosine_similarities.values()),
    marker='o',
    color="#1f77b4",
    linewidth=2,
    markersize=2,
    alpha=0.8,
  )

  ax.set_xlabel(
    xlabel="Layer Index",
    labelpad=10,
  )
  ax.set_ylabel(
    ylabel="Cosine Similarity",
    labelpad=10,
  )
  ax.set_title(
    label=f"Cosine Similarity per Layer between candidate directions and {x2_name}",
    pad=20,
  )
  ax.grid(
    visible=True,
    linestyle='--',
    alpha=0.3,
  )

  fig.tight_layout()

  return fig