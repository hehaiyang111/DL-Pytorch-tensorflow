from torch import nn, optim, from_numpy
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)

# 除了最后一列都是训练数据
x_data = from_numpy(xy[:, 0:-1])
# 最后一列为label
y_data = from_numpy(xy[:, [-1]])

# (759,8) and (759,1)
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmod(self.l1(x))
        out2 = self.sigmod(self.l2(out1))
        y_pred = self.sigmod(self.l3(out2))
        return y_pred

# define our model
model = Model()

# define loss
criterion = nn.BCELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train loop
for epoch in range(1000):
    # Forward pass: compute predicted y by passing x to the model
    y_pred = model(x_data)

    # compute and print loss
    loss = criterion(y_pred, y_data)
    print(f"Epoch: {epoch + 1} | loss: {loss}")

    # update parameters including zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

