# %%

import unittest
from utils import load_model_and_tokenizer

class TestLoadModelAndTokenizer(unittest.TestCase):
  def test_load_model_and_tokenizer(self):
    model, tokenizer = load_model_and_tokenizer(
      models_path="/root/autodl-fs/transformers",
      model_name="huginn-0125",
    )
    
    self.assertIsNotNone(model, "Model should not be None")
    self.assertIsNotNone(tokenizer, "Tokenizer should not be None")

if __name__ == "__main__":
  unittest.main()

# %%