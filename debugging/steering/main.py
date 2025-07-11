# %%

import torch

# %%

# with projection
# i.e. feature amplification or correction

# when input tensor is orthogonal to the feature
feature = torch.tensor([10.0,0.0])
feature = feature / feature.norm(dim=-1)
print("feature:", feature)
input_tensor = torch.tensor([0.0,10.0])
print("input tensor:", input_tensor)
projection = (input_tensor @ feature).unsqueeze(-1).abs() * feature
print("projection:", projection)
input_tensor = input_tensor + projection
print("new input tensor:", input_tensor)

# %%

# with projection
# i.e. feature amplification or correction

# when input tensor is not orthogonal to the feature
feature = torch.tensor([10.0,0.0])
feature = feature / feature.norm(dim=-1)
print("feature:", feature)
input_tensor = torch.tensor([1.0,10.0])
print("input tensor:", input_tensor)
projection = (input_tensor @ feature).unsqueeze(-1).abs() * feature
print("projection:", projection)
input_tensor = input_tensor + projection
print("new input tensor:", input_tensor)

# %%

# without projection
# i.e. feature amplification or correction

# when input tensor is not aligned with the feature
feature = torch.tensor([-10.0,0.0])
feature = feature / feature.norm(dim=-1)
print("feature:", feature)
input_tensor = torch.tensor([-1.0,10.0])
print("input tensor:", input_tensor)
projection = (input_tensor @ feature).unsqueeze(-1).abs() * feature
print("projection:", projection)
input_tensor = input_tensor + projection
print("new input tensor:", input_tensor)

# %%

# feature addition

feature = torch.tensor([10.0,0.0])
feature = feature / feature.norm(dim=-1)
print("feature:", feature)
input_tensor = torch.tensor([-1.0,10.0])
print("input tensor:", input_tensor)
input_tensor = input_tensor + feature
print("new input tensor:", input_tensor)

# %%