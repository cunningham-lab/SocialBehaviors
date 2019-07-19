import torch
import torch.optim as optim
import numpy as np


def linear_model(x, W, b):
    return torch.matmul(x, W) + b


data = torch.tensor(np.arange(12).reshape(4,3), dtype=torch.float)

true_W = torch.tensor(np.arange(3).reshape(3,1)+1, dtype=torch.float)
true_b = torch.tensor([[1],[2],[3],[5]], dtype=torch.float)

target = linear_model(data, true_W, true_b)

W = torch.randn((3, 1), dtype=torch.float, requires_grad=True)
#b = torch.randn((4,1), dtype=torch.float, requires_grad=True)
b = true_b

optimizer = optim.Adam([W], lr=0.001)

criterion = torch.nn.MSELoss()

# clear out the gradients of all Variables
# in this optimizer (i.e. W, b)
for i in np.arange(3000):
    optimizer.zero_grad()
    output = linear_model(data, W, b)
    loss = criterion(output, target)
    loss.backward()
    #print(optimizer.param_groups[0]['lr'])
    optimizer.step()



