import torch

# 条件张量


# 选择值张量
x = torch.tensor([10., 20., 30.])  # 形状为 (3,)
y = torch.tensor([5., 5., 5.])     # 形状为 (3,)
condition = x == 10
# 使用 torch.where
result = torch.where(condition, x, y)

print(result)