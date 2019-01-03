import torch
import numpy as np

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class IrisDataset(Dataset):
    def __init__(self, path, delimiter=','):
        self.__data = np.genfromtxt(path, delimiter=delimiter).astype(np.float32)

    def __getitem__(self, index):
        instance = self.__data[index,:]
        data = torch.from_numpy(instance[:-1])
        label = torch.from_numpy(np.array(instance[-1]).astype(int))

        return data, label

    def __len__(self):
        return self.__data.shape[0]

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

if __name__ == '__main__':
    dataloader = DataLoader(dataset=IrisDataset('iris.data'),
                            batch_size=10,
                            shuffle=True)

    epochs = 50
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        running_loss = 0
        for instances, labels in dataloader:
            optimizer.zero_grad()
            
            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(running_loss / len(dataloader))

    instances, labels = next(iter(dataloader))
    instance = instances[0].view(1, 4)
    label = labels[0].view(1, 1)
    print(torch.exp(model(instance)), label)
