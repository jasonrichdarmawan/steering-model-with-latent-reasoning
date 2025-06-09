import unittest
import sys

from experiments.save_hidden_states.shs_utils import parse_args

class TestParseArgs(unittest.TestCase):
  def setUp(self):
    self.original_argv = sys.argv.copy()
  
  def tearDown(self):
    sys.argv = self.original_argv

  def test_parse_args(self):
    # Simulate command line arguments
    sys.argv = [
      'test_script.py',
      '--models_path', '/path/to/models',
      '--model_name', 'my_model'
    ]
    
    args = parse_args()
    
    self.assertEqual(args['models_path'], '/path/to/models')
    self.assertEqual(args['model_name'], 'my_model')