from matplotlib import pyplot as plt
import numpy as np

def create_effect_per_layer_plot(
  effects: dict[int, list[float]],
  model_name: str,
):
  fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(12, 10),
    facecolor='white'
  )

  max_layer = max(effects.keys())
  all_layers = range(max_layer + 1)

  mean_effects = np.array([
    np.mean(effects.get(layer_index, [0.0]))
    for layer_index in all_layers
  ])
  std_effects = np.array([
    np.std(effects.get(layer_index, [0.0]))
    for layer_index in all_layers
  ])

  ax.fill_between(
    all_layers,
    mean_effects - std_effects,
    mean_effects + std_effects,
    alpha=0.2,
    color="#2E86C1",
  )

  ax.plot(
    all_layers,
    mean_effects,
    color="#2E86C1",
    linewidth=2.5,
    marker='o',
    markersize=4,
  )

  # Remove offset on x-axis and y-axis
  ax.margins(x=0, y=0)

  ax.set_xlabel(
    xlabel="Layer",
    fontsize=18,
    labelpad=12,
    color='black',
  )
  ax.set_ylabel(
    ylabel='Mean KL-Divergence',
    fontsize=18,
    labelpad=12,
    color='black',
  )

  ax.grid(
    visible=True,
    linestyle='--',
    alpha=0.4,
    color='gray',
    which='major',
  )
  
  # Adjust layout to make room for the suptitle
  fig.tight_layout(rect=[0, 0, 1, 0.95])
  fig.suptitle(
    t=model_name,
    fontsize=26,
    y=0.98,
    color="black",
    weight='bold'
  )

  return fig