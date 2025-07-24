# %%

import os
import sys

# To be able to import modules from the shs_utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
  print(f"Adding project root to sys.path: {project_root}")
  sys.path.insert(0, project_root)

# %%

if False:
  from utils import reload_modules
  
  reload_modules(
    project_root=project_root,
  )

from tlp_utils import parse_args
from utils import enable_reproducibility
from utils import SaveHiddenStatesOutput
from utils import SaveHiddenStatesQueryLabel

import torch as t
import einops
import numpy as np
from tqdm import tqdm
from jaxtyping import Float, Int
from eindex import eindex
import matplotlib.pyplot as plt
import contextlib
from typing import TypedDict

# %%

if False:
  print("Programatically setting sys.argv for testing purposes.")
  WORKSPACE_PATH = "/root/autodl-fs"
  sys.argv = [
    'main.py',

    '--device', 'cuda:0',

    '--hidden_states_file_path', f'{WORKSPACE_PATH}/experiments/save_hidden_states/huginn-0125_hidden_states_FIRST_ANSWER_TOKEN.pt',
    
    '--epochs', '1000',

    '--lr', str(1e-4),
    '--weight_decay', str(1e-2), 

    # '--sample_size', '200', 
    '--test_ratio', '0.25',
    '--batch_size', '32',

    # '--moving_average_window_size_step', '50',
    # '--moving_average_window_size_epoch', '1',

    '--output_dir', f'{WORKSPACE_PATH}/experiments/train_linear_probes',
    '--checkpoint_freq', '50',
    '--use_early_stopping', 
    '--early_stopping_patience', '10',
  ]

args = parse_args()

print("Parsed arguments:")
print("#" * 60)
for key, value in args.items():
  print(f"{key}: {value}")

# %%

print("Setting deterministic behavior for reproducibility.")
enable_reproducibility()

# %%

print("Loading hidden states from file.")
hidden_states: SaveHiddenStatesOutput = t.load(
  args['hidden_states_file_path'],
  map_location='cpu',
  weights_only=False,
)

# %%

x = t.stack(
  [t.stack(hs) for _, hs in hidden_states['hidden_states'].items()],
)
x = einops.rearrange(
  x,
  'n_layers n_samples n_embd -> n_samples n_layers n_embd',
)

label_to_idx = {
  SaveHiddenStatesQueryLabel.REASONING.value: 0,
  SaveHiddenStatesQueryLabel.MEMORIZING.value: 1,
}
y = t.tensor(
  [label_to_idx[label] for label in hidden_states['labels']],
)

num_samples = args['sample_size'] or x.shape[0]
test_size = int(num_samples * args['test_ratio'])
indices = t.randperm(num_samples)
train_indices = indices[test_size:]
test_indices = indices[:test_size]

dtype = t.float64 # for numerical stability

x_train = x[train_indices].to(device=args['device'], dtype=dtype)
y_train = y[train_indices].to(device=args['device'])

x_test = x[test_indices].to(device=args['device'], dtype=dtype)
y_test = y[test_indices].to(device=args['device'])

del x
del y
del label_to_idx

# %%

del hidden_states

# %%

def compute_epoch_average(
  losses: list[float], 
  n_epochs: int
):
  if not losses:
    return []
  epoch_size = len(losses) // n_epochs
  return [
    np.mean(losses[i * epoch_size:(i + 1) * epoch_size])
    for i in range(n_epochs)
  ]

# %%

class LinearProbeTrainerArgs:
  def __init__(
    self,
    device: t.device,
    n_epochs: int, 
    lr: float, 
    weight_decay: float,
    x_train: Float[t.Tensor, "batch_size n_layers n_embd"], 
    y_train: Float[t.Tensor, "batch_size"], 
    x_test: Float[t.Tensor, "test_size n_layers n_embd"], 
    y_test: Float[t.Tensor, "test_size"], 
    batch_size: int, 
    output_dir: str | None = None, 
    checkpoint_freq: int = 50, 
    use_early_stopping: bool = False,
    early_stopping_patience: int = 10,
  ):
    self.device = device

    self.n_epochs = n_epochs
    self.lr = lr
    self.weight_decay = weight_decay

    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

    self.batch_size = batch_size
    self.train_size = x_train.shape[0]  # n_samples
    self.test_size = x_test.shape[0]    # n_samples

    self.n_layers = x_train.shape[1]  # n_layers
    self.n_embd = x_train.shape[2]    # n_embd
    self.options = len(y_train.unique())
    self.dtype = x_train.dtype

    self.output_dir = output_dir
    self.checkpoint_freq = checkpoint_freq

    self.use_early_stopping = use_early_stopping
    self.early_stopping_patience = early_stopping_patience

  def setup_linear_probe(
    self,
  ):
    linear_probe = t.randn(
      self.n_layers,
      self.n_embd,
      self.options,
      dtype=self.dtype,
      device=self.device,
    ) / t.sqrt(t.tensor(
      self.n_embd,
      dtype=self.dtype, 
      device=self.device,
    ))
    linear_probe.requires_grad = True
    return linear_probe

class LinearProbeTrainerResult(TypedDict):
  config: dict
  logs: list[dict[str, float]]

class LinearProbeTrainer:
  def __init__(
    self,
    args: LinearProbeTrainerArgs,
  ):
    self.args = args
    self.linear_probe = args.setup_linear_probe()

  def train(self):
    self.current_epoch = 0
    self.train_step = 0
    self.test_step = 0

    config = {
      'epochs': self.args.n_epochs,
      'lr': self.args.lr,
      'train_size': self.args.train_size,
      'batch_size': self.args.batch_size,
    },
    self.results: LinearProbeTrainerResult = {
      'config': config,
      'logs': [],
    }

    self.optimizer = t.optim.AdamW(
      [self.linear_probe],
      lr=self.args.lr,
      weight_decay=self.args.weight_decay,
    )

    self.best_layer_index = 0

    best_epoch = 0
    best_train_loss = float('inf')
    best_test_loss = float('inf')
    epochs_since_best = 0

    progress_bar = tqdm(range(self.args.n_epochs))
    for epoch in progress_bar:
      full_train_indices = self.shuffle_training_indices()

      for indices in full_train_indices:
        train_total_loss, train_data = self.training_step(indices)

        train_total_loss.backward()

        """
        Gradient clipping
        The norm is computed over all gradients of a linear probe parameters,
        and if the norm exceeds `max_norm`, the gradients are scaled down
        ```python
        linear_probe = t.arange(0, 2 * 10, dtype=t.float32).reshape(2,10)
        linear_probe.requires_grad = True
        optimizer = t.optim.SGD([linear_probe], lr=0.1)

        loss = linear_probe[0].sum()
        # loss += (linear_probe[1] * 2).sum()
        loss.backward()

        print("Parameter before step:")
        print("linear_probe[0]:", linear_probe[0])
        print("linear_probe[1]:", linear_probe[1])

        print("Gradients before clipping:")
        print("linear_probe.grad[0]:", linear_probe.grad[0])
        print("linear_probe.grad[1]:", linear_probe.grad[1])
        grad = linear_probe.grad[0]
        norm = grad.norm()
        if norm > 1.0:
          linear_probe.grad[0].mul_(1.0 / norm)
        print("Gradients after clipping:")
        print("linear_probe.grad[0]:", linear_probe.grad[0])
        print("linear_probe.grad[1]:", linear_probe.grad[1])

        optimizer.step()

        print("Parameter after step")
        print("linear_probe[0]:", linear_probe[0])
        print("linear_probe[1]:", linear_probe[1])
        ```
        """
        with t.no_grad():
          for i in range(self.linear_probe.shape[0]):
            grad = self.linear_probe.grad[i]
            norm = grad.norm()
            if norm > 1.0:
              self.linear_probe.grad[i].mul_(1.0 / norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.results['logs'].append({
          'data': {
            **train_data,
            'train_step': self.train_step,
          },
        })

        self.train_step += 1

      # Testing step
      full_test_indices = self.get_test_indices()
      for indices in full_test_indices:
        _, test_data = self.testing_batch(indices)

        self.results['logs'].append({
          'data': {
            **test_data,
            'test_step': self.test_step,
          },
        })

        self.test_step += 1

      # Save checkpoint
      if self.args.output_dir and epoch > 0 and epoch % self.args.checkpoint_freq == 0 or epoch == self.args.n_epochs - 1:
        checkpoint_path = os.path.join(
          self.args.output_dir,
          f'checkpoint_epoch_{epoch}.pt',
        )
        self.save_checkpoint(checkpoint_path=checkpoint_path)
      
      # Check for best test loss
      start_train_step = self.train_step - len(full_train_indices)
      end_train_step = self.train_step
      start_test_step = self.test_step - len(full_test_indices)
      end_test_step = self.test_step

      layerwise_train_losses_epoch = t.tensor(
        [
          [
            log['data'][f'train/loss_{layer_index}'] 
            for log in self.results['logs']
            if f'train/loss_{layer_index}' in log['data'] 
            and start_train_step <= log['data']['train_step'] < end_train_step
          ]
          for layer_index in range(self.args.n_layers)
        ], 
        device=self.args.device,
      ).mean(dim=1)
      layerwise_test_losses_epoch = t.tensor(
        [
          [
            log['data'][f'validate/loss_{layer_index}'] 
            for log in self.results['logs']
            if f'validate/loss_{layer_index}' in log['data'] 
            and start_test_step <= log['data']['test_step'] < end_test_step
          ]
          for layer_index in range(self.args.n_layers)
        ], 
        device=self.args.device, 
      ).mean(dim=1)
      best_layer_index_epoch = t.argmin(layerwise_test_losses_epoch)
      best_train_loss_epoch = layerwise_train_losses_epoch[best_layer_index_epoch]
      best_test_loss_epoch = layerwise_test_losses_epoch[best_layer_index_epoch]
      if best_test_loss > best_test_loss_epoch:
        best_epoch = epoch
        best_layer_index = best_layer_index_epoch
        best_train_loss = best_train_loss_epoch
        best_test_loss = best_test_loss_epoch
        if self.args.output_dir:
          checkpoint_path = os.path.join(
            self.args.output_dir,
            f'best_checkpoint.pt',
          )
          self.save_checkpoint(
            checkpoint_path=checkpoint_path,
            layer_index=best_layer_index_epoch,
            epoch=epoch,
          )
        epochs_since_best = 0
      else:
        epochs_since_best += 1

      # Early stopping
      if self.args.use_early_stopping and epochs_since_best == self.args.early_stopping_patience:
        print(f"Early stopping triggered after {self.args.early_stopping_patience} epochs without improvement.")
        break

      progress_bar.set_description(f"Epoch {epoch}/{self.args.n_epochs} Layer: {best_layer_index_epoch} Train Loss: {best_train_loss_epoch.item():.4f} Test Loss: {best_test_loss_epoch.item():.4f} Best Epoch: {best_epoch} Layer: {best_layer_index} Train Loss: {best_train_loss.item():.4f} Test Loss: {best_test_loss.item():.4f}")

      self.current_epoch += 1

  def shuffle_training_indices(self):
    n_indices = self.args.train_size - (self.args.train_size % self.args.batch_size)
    full_train_indices = t.randperm(self.args.train_size)[:n_indices]
    full_train_indices = einops.rearrange(
      full_train_indices,
      '(b s) -> b s',
      b=n_indices // self.args.batch_size,
      s=self.args.batch_size,
    )
    return full_train_indices
  
  def get_test_indices(self):
    n_indices = self.args.test_size - (self.args.test_size % self.args.batch_size)
    full_train_indices = t.arange(0, n_indices)
    full_train_indices = einops.rearrange(
      full_train_indices,
      '(b s) -> b s',
      b=n_indices // self.args.batch_size,
      s=self.args.batch_size,
    )
    return full_train_indices
  
  def training_step(
    self,
    indices: Int[t.Tensor, "batch_size"],
  ):
    x_batch = self.args.x_train[indices]
    y_batch = self.args.y_train[indices]

    return self.step_core(
      x_batch=x_batch,
      y_batch=y_batch,
      no_grad=False,
    )
  
  def testing_batch(
    self,
    indices: Int[t.Tensor, "n_samples"]
  ):
    x_batch = self.args.x_test[indices]
    y_batch = self.args.y_test[indices]

    return self.step_core(
      x_batch=x_batch,
      y_batch=y_batch,
      no_grad=True,
    )

  def save_checkpoint(
    self,
    checkpoint_path: str,
    **kwargs,
  ):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    t.save(
      {
        'linear_probe': self.linear_probe,
        'optimizer_state_dict': self.optimizer.state_dict(),
        'results': self.results,
        **kwargs,
      }, 
      checkpoint_path, 
    )
  
  def step_core(
    self,
    x_batch: Float[t.Tensor, "batch_size n_layers n_embd"],
    y_batch: Int[t.Tensor, "batch_size"],
    no_grad: bool = False,
  ):
    ctx = t.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
      probe_logits = einops.einsum(
        x_batch,
        self.linear_probe,
        'n_samples n_layers n_embd, n_layers n_embd n_options -> n_layers n_samples n_options'
      )
      probe_logprobs = probe_logits.log_softmax(dim=-1)

      correct_probe_logprobs = eindex(
        probe_logprobs,
        y_batch,
        'n_layers n_samples [n_samples]',
      ) # shape (n_layers, n_samples)

      losses = -correct_probe_logprobs.mean(dim=-1)
      total_loss = losses.sum()

    label = 'train' if not no_grad else 'validate'
    data = {
      f'{label}/total_loss': total_loss.item(),
      **{f"{label}/loss_{layer_index}": losses[layer_index].item() 
         for layer_index in range(losses.shape[0])},
    }

    return total_loss, data

trainer_args = LinearProbeTrainerArgs(
  device=args['device'],
  n_epochs=args['epochs'],
  lr=args['lr'],
  weight_decay=args['weight_decay'],
  batch_size=args['batch_size'],
  x_train=x_train,
  y_train=y_train,
  x_test=x_test,
  y_test=y_test,
  output_dir=args.get('output_dir', None),
  checkpoint_freq=args.get('checkpoint_freq', 50),
  use_early_stopping=args['use_early_stopping'],
  early_stopping_patience=args.get('early_stopping_patience', 10),
)

trainer = LinearProbeTrainer(
  args=trainer_args,
)
trainer.train()

# %%

print("Plotting training losses and per-layer losses...")

def compute_moving_average(
  data: list[float],
  window_size: int,
) -> list[float]:
  window = np.ones(window_size) / window_size
  smoothed_data = np.convolve(a=data, v=window, mode='valid')
  return smoothed_data

plots_per_row = 3
layers_per_plot = 4
n_layers = trainer_args.n_layers
n_plots = 2 + (n_layers + layers_per_plot - 1) // layers_per_plot
n_rows = (n_plots + plots_per_row - 1) // plots_per_row
n_cols = plots_per_row

window_size = args['moving_average_window_size_step'] or 1

fig, axes = plt.subplots(
  nrows=n_rows,
  ncols=n_cols,
  figsize=(plots_per_row * 6.48, n_rows * 4.8),
)
axes = axes.flatten()

logs = trainer.results['logs']

train_steps = [log['data']['train_step'] for log in logs if 'train_step' in log['data']]
train_losses = [log['data']['train/total_loss'] for log in logs if 'train/total_loss' in log['data']]

smoothed_train_steps = train_steps[window_size - 1:]  # Adjust steps to match smoothed losses
smoothed_train_losses = compute_moving_average(
  data=train_losses, 
  window_size=window_size
)

axes[0].plot(
  smoothed_train_steps, 
  smoothed_train_losses, 
  label='Train Loss'
)
axes[0].set_title('Train Losses')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

best_layer_index = trainer.best_layer_index
best_train_losses = [
  log['data'][f'train/loss_{best_layer_index}'] 
  for log in logs if f'train/loss_{best_layer_index}' in log['data']
]
smoothed_best_train_losses = compute_moving_average(
  data=best_train_losses,
  window_size=window_size,
)

axes[1].plot(
  smoothed_train_steps,
  smoothed_best_train_losses,
  label='Layer {best_layer_index}',
)
axes[1].set_title('Best Layer Train Losses')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

for plot_idx in range(2, n_plots):
  ax = axes[plot_idx]

  start_layer_index = (plot_idx - 1) * layers_per_plot
  end_layer_index = min(start_layer_index + layers_per_plot, n_layers)
  for layer_index in range(start_layer_index, end_layer_index):
    train_losses_per_layer = [
      log['data'][f'train/loss_{layer_index}'] for log in logs 
      if f'train/loss_{layer_index}' in log['data']
    ]
    smoothed_train_losses_per_layer = compute_moving_average(
      data=train_losses_per_layer,
      window_size=window_size,
    )
    ax.plot(
      smoothed_train_steps,
      smoothed_train_losses_per_layer,
      label=f'Layer {layer_index}',
    )
    ax.set_title(f'Layer {start_layer_index} to {end_layer_index - 1} Train Losses')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

# Hide any unused axes
for i in range(n_plots, len(axes)):
  axes[i].axis('off')

# %%

print("Saving training losses plot...")
if args['output_dir']:
  os.makedirs(args['output_dir'], exist_ok=True)
  plot_path = os.path.join(args['output_dir'], 'train_losses.png')
  fig.savefig(plot_path, bbox_inches='tight')
  print(f"Plot saved to {plot_path}")

# %%

print("Comparing train and test losses...")

plots_per_row = 3
layers_per_plot = 4
n_layers = trainer_args.n_layers
n_plots = 2 + (n_layers + layers_per_plot - 1) // layers_per_plot
n_rows = (n_plots + plots_per_row - 1) // plots_per_row
n_cols = plots_per_row

window_size = args['moving_average_window_size_epoch'] or 1

fig, axes = plt.subplots(
  nrows=n_rows,
  ncols=n_cols,
  figsize=(plots_per_row * 6.48, n_rows * 4.8),
)
axes = axes.flatten()

logs = trainer.results['logs']

train_steps = [log['data']['train_step'] for log in logs if 'train_step' in log['data']]
train_losses = [log['data']['train/total_loss'] for log in logs if 'train/total_loss' in log['data']]
test_steps = [log['data']['test_step'] for log in logs if 'test_step' in log['data']]
test_losses = [log['data']['validate/total_loss'] for log in logs if 'validate/total_loss' in log['data']]

n_epochs = trainer.current_epoch
epochs = list(range(n_epochs))
train_losses_per_epoch = compute_epoch_average(
  losses=train_losses,
  n_epochs=n_epochs
)
test_losses_per_epoch = compute_epoch_average(
  losses=test_losses, 
  n_epochs=n_epochs
)

smoothed_train_losses_per_epoch = compute_moving_average(
  data=train_losses_per_epoch,
  window_size=window_size,
)
smoothed_test_losses_per_epoch = compute_moving_average(
  data=test_losses_per_epoch,
  window_size=window_size,
)
smoothed_epochs = epochs[window_size - 1:]  # Adjust epochs to match smoothed losses

axes[0].plot(
  smoothed_epochs, 
  smoothed_train_losses_per_epoch, 
  label='Train Loss'
)
axes[0].plot(
  smoothed_epochs, 
  smoothed_test_losses_per_epoch, 
  label='Test Loss',
)
axes[0].set_title('Train Losses')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

best_layer_index = trainer.best_layer_index
best_train_losses = compute_epoch_average(
  losses=[
    log['data'][f'train/loss_{best_layer_index}'] 
    for log in logs if f'train/loss_{best_layer_index}' in log['data']
  ],
  n_epochs=n_epochs,
)
best_test_losses = compute_epoch_average(
  losses=[
    log['data'][f'validate/loss_{best_layer_index}'] 
    for log in logs if f'validate/loss_{best_layer_index}' in log['data']
  ],
  n_epochs=n_epochs,
)
smoothed_best_train_losses = compute_moving_average(
  data=best_train_losses,
  window_size=window_size,
)
axes[1].plot(
  smoothed_epochs,
  smoothed_best_train_losses,
  label=f'Layer {best_layer_index} Train Loss',
)
axes[1].plot(
  smoothed_epochs,
  best_test_losses,
  label=f'Layer {best_layer_index} Test Loss',
)
axes[1].set_title('Best Layer Train and Test Losses')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

for plot_idx in range(2, n_plots):
  ax = axes[plot_idx]

  start_layer_index = (plot_idx - 1) * layers_per_plot
  end_layer_index = min(start_layer_index + layers_per_plot, n_layers)
  for layer_index in range(start_layer_index, end_layer_index):
    train_losses_per_layer = [
      log['data'][f'train/loss_{layer_index}'] for log in logs 
      if f'train/loss_{layer_index}' in log['data']
    ]
    test_losses_per_layer = [
      log['data'][f'validate/loss_{layer_index}'] for log in logs 
      if f'validate/loss_{layer_index}' in log['data']
    ]

    train_losses_per_epoch_per_layer = compute_epoch_average(
      losses=train_losses_per_layer,
      n_epochs=n_epochs,
    )
    test_losses_per_epoch_per_layer = compute_epoch_average(
      losses=test_losses_per_layer,
      n_epochs=n_epochs,
    )

    smoothed_train_losses_per_epoch_per_layer = compute_moving_average(
      data=train_losses_per_epoch_per_layer,
      window_size=window_size,
    )
    smoothed_test_losses_per_epoch_per_layer = compute_moving_average(
      data=test_losses_per_epoch_per_layer,
      window_size=window_size,
    )
    ax.plot(
      smoothed_epochs,
      smoothed_train_losses_per_epoch_per_layer,
      label=f'Layer {layer_index} Train Loss',
    )
    ax.plot(
      smoothed_epochs,
      smoothed_test_losses_per_epoch_per_layer,
      label=f'Layer {layer_index} Test Loss',
    )
    ax.set_title(f'Layer {start_layer_index} to {end_layer_index - 1} Train Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

# Hide any unused axes
for i in range(n_plots, len(axes)):
  axes[i].axis('off')

# %%

print("Saving train and test losses plot...")
if args['output_dir']:
  os.makedirs(args['output_dir'], exist_ok=True)
  plot_path = os.path.join(args['output_dir'], 'train_test_losses.png')
  fig.savefig(plot_path, bbox_inches='tight')
  print(f"Plot saved to {plot_path}")

# %%
