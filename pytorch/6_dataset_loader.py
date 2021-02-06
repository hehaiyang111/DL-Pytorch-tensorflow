from torch.utils.data import DataLoader, Dataset
from torch import from_numpy, tensor
import numpy as np

class DiabetersDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt("./data/diabetes.csv.gz", delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:,0:-1])
        self.y_data = from_numpy(xy[:,[-1]])
    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

dataSets = DiabetersDataset()

train_loader = DataLoader(dataset=dataSets,
                          batch_size=32,
                          shuffle=True,
                          num_workers=0)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        inputs, labels = tensor(inputs), tensor(labels)

        print(f'Epoch: {i} | Inputs {inputs.data} | Labels {labels.data}')

