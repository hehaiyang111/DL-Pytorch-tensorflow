# celebrate the data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1 # guess the w

# our model for the forward pass
def forward(x):
    return x * w

# loss function
def loss(y_pred, y):
    return (y_pred - y) * (y_pred - y)

def gradient(x, y):
    return 2 * x * (w * x - y)



for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        x_pred = forward(x_val)
        # loss
        l = loss(x_pred, y_val)
        # gradient
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        # print("\tgrad: ", x_val, y_val, round(grad, 2))
    print("progress", epoch, "W=", round(w,2), "loss=", round(l, 2))


