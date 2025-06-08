# %%

from utils.use_deterministic_algorithms import use_deterministic_algorithms
from utils.parse_args import parse_args
from utils.load_model_and_tokenizer import load_model_and_tokenizer

use_deterministic_algorithms()

args = parse_args()

model, tokenizer = load_model_and_tokenizer(
  model_path=args['models_path'],
  model_name=args['model_name'],
)

# %%
