from jaxtyping import Float
import torch
from torch import Tensor
from matplotlib import pyplot as plt

def compute_cosine_similarities(
  x1: dict[int, Float[Tensor, "n_embd"]],
  x2: Float[Tensor, "block_size n_embd"],
):
  cosine_similarities: dict[int, float] = {}
  for layer_index, item in x1.items():
    cosine_similarity = torch.cosine_similarity(
      x1=item.to(device=x2.device),
      x2=x2,
      dim=-1,
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