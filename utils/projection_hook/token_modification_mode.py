from enum import Enum

class TokenModificationMode(Enum):
  ALL_TOKENS = "ALL_TOKENS"
  LAST_TOKEN = "LAST_TOKEN"

  def __str__(self):
    return self.value