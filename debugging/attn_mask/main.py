# %%

import torch
from torch import nn
import torch.nn.functional as F
import math

PAD_TOKEN_ID = 0
HELLO_TOKEN_ID = 1
WORLD_TOKEN_ID = 2

num_embeddings = 3
n_head = 2
n_embd = 4
head_dim = n_embd // n_head

input_ids = torch.tensor([
  [PAD_TOKEN_ID, HELLO_TOKEN_ID],
  [HELLO_TOKEN_ID, WORLD_TOKEN_ID],
])
print("input_ids.shape:", input_ids.shape)
print(f"input_ids:\n{input_ids}")

attention_mask = (input_ids != PAD_TOKEN_ID).long()
print("attention_mask.shape:", attention_mask.shape)

wte = nn.Embedding(
  num_embeddings=num_embeddings,
  embedding_dim=n_embd,
  padding_idx=PAD_TOKEN_ID,
)
with torch.no_grad():
  wte.weight.copy_(
    torch.arange(
      end=wte.weight.numel(),
      dtype=wte.weight.dtype
    ).reshape(shape=wte.weight.shape)
  )
print(f"wte.weight:\n{wte.weight}")

input_embeds = wte(input_ids)
print("input_embeds.shape:", input_embeds.shape)
print(f"input_embeds:\n{input_embeds}")

Wqkv = nn.Linear(
  in_features=n_embd,
  out_features=n_embd * 3,
  bias=False,
)
with torch.no_grad():
  Wqkv.weight.copy_(
    torch.arange(
      end=Wqkv.weight.numel(),
      dtype=Wqkv.weight.dtype
    ).reshape(shape=Wqkv.weight.shape)
  )
print("Wqkv.weight.shape:", Wqkv.weight.shape)
print(f"Wqkv.weight:\n{Wqkv.weight}")

B, S, E = input_embeds.shape

qkv = Wqkv(input_embeds)
print("qkv.shape:", qkv.shape)
print(f"qkv:\n{qkv}")

q, k, v = qkv.split(split_size=n_embd, dim=-1)
q = q.view(B, S, n_head, head_dim).transpose(1, 2)
k = k.view(B, S, n_head, head_dim).transpose(1, 2)
v = v.view(B, S, n_head, head_dim).transpose(1, 2)
print("q.shape:", q.shape)
print(f"q:\n{q}")
print("k.shape:", k.shape)
print(f"k:\n{k}")
print(f"v.shape:", v.shape)
print(f"v:\n{v}")

# Manual Attention Calculation
print("=" * 50)
print("Manual Attention Calculation")
att = (q @ k.transpose(-2, -1))
print(f"QK^T:\n{att}")
att = att / math.sqrt(k.size(-1))
print(f"QK^T / \sqrt{{d_k}}:\n{att}")
padding_mask = attention_mask.unsqueeze(dim=1).unsqueeze(dim=2) == 0
print("padding_mask.shape:", padding_mask.shape)
print(f"padding_mask:\n{padding_mask}")
causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool)) == 0
print("causal_mask.shape:", causal_mask.shape)
print(f"causal_mask:\n{causal_mask}")
mask = padding_mask | causal_mask
print("mask.shape:", mask.shape)
print(f"mask:\n{mask}")
att = att.masked_fill(
  mask=mask,
  # `True` indicates positions to mask out
  # Masking logic: mask out positions where the key token is padding.
  value=float("-inf"),
)
print(f"QK^T / \sqrt{{d_k}} with attention mask:\n{att}")

att = att.softmax(dim=-1)
print(f"Softmax(QK^T / \sqrt{{d_k}}):\n{att}")
y = att @ v
print("y.shape:", y.shape)
print(f"y:\n{y}")

# Flash Attention
print("=" * 50)
print("Flash Attention Calculation")
padding_mask = attention_mask.unsqueeze(dim=1).unsqueeze(dim=2) == 1
print("padding_mask.shape:", padding_mask.shape)
print(f"padding_mask:\n{padding_mask}")
causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool)) == 1
print("causal_mask.shape:", causal_mask.shape)
print(f"causal_mask:\n{causal_mask}")
mask = padding_mask & causal_mask
print("mask.shape:", mask.shape)
print(f"mask:\n{mask}")
y = F.scaled_dot_product_attention(
  query=q,
  key=k,
  value=v,
  attn_mask=padding_mask,
  # The `attn_mask` parameter expects
  # `True` for positions to keep and
  # `False` for positios to ignore
  is_causal=True,
  # PyTorch 2.7.1+cu126 does not throw
  # an error if both `attn_mask` and `is_causal` are set.
  # and it seems to combine the `attn_mask` with the `causal_mask`.
  # However, it is recommended to set `is_causal` to `False`
  # and combine the masks manually.
)
print("y.shape:", y.shape)
print(f"y:\n{y}")

# %%
