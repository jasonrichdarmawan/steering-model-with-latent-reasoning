import torch
import random
import numpy as np

def enable_reproducibility():
  """
  Warning: Deterministic operations are often slower
  than nondeterministic operations, so single-run
  performance may decrease for your model. However,
  determinism may save time in development by
  facilitating experimentation, debugging, and
  regression testing.

  Reference: https://docs.pytorch.org/docs/stable/notes/randomness.html#reproducibility
  """
  
  # Seed the RNG for all devices (both CPU and CUDA)
  # Some PyTorch operations may use random numbers
  # internally. Consequently, calling it multiple
  # times back-to-back with the same input arguments
  # may give different results.
  torch.manual_seed(seed=0)
  
  random.seed(a=0)

  # Seed the global NumPy RNG
  # However, some applications and libraries may
  # use NumPy Random Generator objects, not the
  # global RNG (https://numpy.org/doc/stable/reference/random/generator.html), 
  # and those will need to be seeded consistently 
  # as well.
  np.random.seed(seed=0)

  # causes cuDNN to deterministically select an algorithm,
  # possibly at the cost of reduced performance.
  torch.backends.cudnn.benchmark = False

  # Avoiding nondeterministic algorithms
  torch.use_deterministic_algorithms(
    mode=True,
    warn_only=True,
  )
  
  # CUDA convolution determinism
  # While disabling CUDA convolution benchmarking
  # ensures that CUDA selects the same algorithm
  # each time an application is run, that
  # algorithm itself may be nondeterministic,
  # unless either `torch.use_deterministic_algorithms(True)``
  # or `torch.backends.cudnn.deterministic = True`
  # is set. The latter setting controls only this behavior,
  # unlike `torch.use_deterministic_algorithms(True)`
  # which will make other PyTorch operations
  # behave deterministically as well.
  torch.backends.cudnn.deterministic = True