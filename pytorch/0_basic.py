import numpy as np
import matplotlib.pyplot as plt

# celebrate the data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x):
    return x * w

# loss function
def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)

# list of weights/Mean square Error(MSE) for each input
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0

    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(y_pred_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)

    # Now compute the Mean squared error(MSE) of each
    print("MSE=", l_sum / len(x_data))
    w_list.append(w)
    mse_list.append(l_sum / len(x_data))

# Plot it all
plt.xlabel('w')
plt.ylabel('loss')
plt.plot(w_list, mse_list)
plt.show()
