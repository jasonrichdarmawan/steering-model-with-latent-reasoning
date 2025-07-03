import unittest

import torch

from utils import enable_reproducibility

class TestUseDeterministicAlgorithms(unittest.TestCase):
  def test_use_deterministic_algorithms(self):
    """
    TODO: real test
    """
    enable_reproducibility()

    def run_nondeterministic_code():
      torch.manual_seed(0)
      return torch.rand(2, 2)

    a = run_nondeterministic_code()
    b = run_nondeterministic_code()

    self.assertEqual(
      a.tolist(), b.tolist(),
      "The outputs of the nondeterministic code should be equal after using deterministic algorithms.")