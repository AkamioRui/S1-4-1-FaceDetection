import torch
import torchmetrics


# a = torch.tensor([[1,2],[3,4]])
# for x in a:
#     print(x)

# a = torch.zeros([2,2])
# print(f"{a.shape=}")
# print(f"{a.shape[0:2]=}")
# print(f"{torch.zeros(a.shape[0:1])=}")

a = torch.tensor([[1,2],[3,4]])
b = a * torch.tensor([[1,-1]])
print(f"{b=}")