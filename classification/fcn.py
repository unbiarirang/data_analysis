import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

D = 178
K = 5

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

        self.conv1  = nn.Conv1d(1, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2  = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3  = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.gap = nn.AvgPool1d(kernel_size=D)
        self.fc= nn.Linear(128, K)

    def forward(self, x):
        x = x.reshape(-1, 1, D)

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = F.relu(self.bn3(out))

#        out = self.gap(out).reshape(1, -1)
        out = self.gap(out).reshape(-1, 128)
        out = self.fc(out)

        return F.softmax(out, dim=1)
