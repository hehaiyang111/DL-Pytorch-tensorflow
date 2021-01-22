import torch

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = torch.tensor([1.0], requires_grad=True)

# forward process
def forward(x):
    return x * w

def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # forward process
        y_pred = forward(x_val)
        # compute loss
        l = loss(y_pred, y_val)
        # back propagation
        l.backward()
        print("\tgrad", x_val, y_val, w.grad.item(), w.data.item())
        w.data = w.data - 0.01 * w.grad.item()
        w.grad.data.zero_()

    print(f"Epoch: {epoch} | loss : {l.item()}")
