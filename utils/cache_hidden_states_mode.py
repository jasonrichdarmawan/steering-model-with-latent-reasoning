from enum import Enum

class CacheHiddenStatesMode(Enum):
  FIRST_ANSWER_TOKEN = "FIRST_ANSWER_TOKEN"
  ALL_TOKENS = "ALL_TOKENS"
  """
  Caching all tokens will take too much disk space
  to store, RAM/VRAM to compute
  """

  def __str__(self):
    return self.value