import torch.nn as nn
import torch.nn.functional as F


# create the models
class NetworkNorm(nn.Module):
    def __init__(self, x_indices, y_indices, SIZE=512, DROPOUT=0.2):
        super(NetworkNorm, self).__init__()
        self.fc1 = nn.Linear(x_indices, SIZE)
        self.fc2 = nn.Linear(SIZE, SIZE // 2)
        self.fc3 = nn.Linear(SIZE // 2, y_indices - 1)
        self.DROPOUT = DROPOUT

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.DROPOUT)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.DROPOUT)
        x = self.fc3(x)

        # can only be positive
        x = F.relu(x)
        return x


class NetworkTotal(nn.Module):
    def __init__(self, x_indices, SIZE=512, DROPOUT=0.2):
        super(NetworkTotal, self).__init__()
        self.fc1 = nn.Linear(x_indices, SIZE)
        self.fc2 = nn.Linear(SIZE, SIZE // 2)
        self.fc3 = nn.Linear(SIZE // 2, 1)
        self.DROPOUT = DROPOUT

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.DROPOUT)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.DROPOUT)
        x = self.fc3(x)

        # can only be positive
        x = F.sigmoid(x)
        return x
