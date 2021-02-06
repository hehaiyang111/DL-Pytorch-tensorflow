
'''
    写在前面：
    1. design your model using class
    2. construct the loss and optimizer(using pytorch api)
    3. traing cycle (forward, backward, update)
'''
import torch
from torch import nn
from torch import tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])

# 1. define the network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# our model
model = Model()

# 2. construct loss optimizer
mseloss = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. train(compute loss , update parameters)
for epoch in range(1000):
    # 1) forward pass
    y_pred = model(x_data)
    # 2) compute and print loss
    loss = mseloss(y_pred, y_data)
    print(f"Epoch: {epoch} | loss: {loss.item()}")

    # update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# test
test_Data = tensor([[4.0]])
y_pred = model(test_Data)
print(y_pred.data[0][0].item())
