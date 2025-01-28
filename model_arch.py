import torch.nn as nn
import torch.nn.functional as F


# create the models
class NetworkNorm(nn.Module):
    def __init__(self, x_indices, y_indices, SIZE=128, DROPOUT=0.1):
        super(NetworkNorm, self).__init__()
        self.fc1 = nn.Linear(x_indices, SIZE)
        self.fc2 = nn.Linear(SIZE, SIZE)
        self.fc3 = nn.Linear(SIZE, SIZE)
        self.fc4 = nn.Linear(SIZE, y_indices - 1)

        self.norm1 = nn.LayerNorm(SIZE)
        self.norm2 = nn.LayerNorm(SIZE)
        self.norm3 = nn.LayerNorm(SIZE)

        self.DROPOUT = DROPOUT

    def forward(self, x):
        x = self.norm1(self.fc1(x))
        x = F.mish(x)
        x = F.dropout(x, p=self.DROPOUT)
        x = self.norm2(self.fc2(x))
        x = F.mish(x)
        x = F.dropout(x, p=self.DROPOUT)
        x = self.norm3(self.fc3(x))
        x = F.mish(x)
        x = F.dropout(x, p=self.DROPOUT)
        x = self.fc4(x)

        x = F.softmax(x, dim=1)
        return x


class NetworkTotal(nn.Module):
    def __init__(self, x_indices, y_indices, SIZE=128, DROPOUT=0.1):
        super(NetworkTotal, self).__init__()
        self.y_indices = y_indices
        self.fc1 = nn.Linear(x_indices, SIZE)
        self.fc2 = nn.Linear(SIZE, SIZE)
        self.fc3 = nn.Linear(SIZE, SIZE)
        self.fc4 = nn.Linear(SIZE, 1)

        self.norm1 = nn.LayerNorm(SIZE)
        self.norm2 = nn.LayerNorm(SIZE)
        self.norm3 = nn.LayerNorm(SIZE)

        self.DROPOUT = DROPOUT

    def forward(self, x):
        x = self.norm1(self.fc1(x))
        x = F.mish(x)
        x = F.dropout(x, p=self.DROPOUT)
        x = self.norm2(self.fc2(x))
        x = F.mish(x)
        x = F.dropout(x, p=self.DROPOUT)
        x = self.norm3(self.fc3(x))
        x = F.mish(x)
        x = F.dropout(x, p=self.DROPOUT)
        x = self.fc4(x)

        # can only be positive
        x = F.sigmoid(x)
        return x
