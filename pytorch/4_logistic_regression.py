import torch

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = Model()


criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_data_1 = torch.tensor([[1.0]])
print(model(test_data_1).data[0][0] > 0.5)
test_data_2 = torch.tensor([[8.0]])
print(model(test_data_2).data[0][0] > 0.5)
