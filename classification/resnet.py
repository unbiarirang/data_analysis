import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

D = 178
K = 5
n_feature_maps = 64

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # BLOCK1
        self.conv1_1  = nn.Conv1d(1, n_feature_maps, kernel_size=7, stride=1, padding=3)
        self.bn1_1 = nn.BatchNorm1d(n_feature_maps)

        self.conv1_2  = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=5, stride=1, padding=2)
        self.bn1_2 = nn.BatchNorm1d(n_feature_maps)

        self.conv1_3  = nn.Conv1d(n_feature_maps, n_feature_maps, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm1d(n_feature_maps)

        self.shortcut1 = nn.Conv1d(1, n_feature_maps, kernel_size=1, stride=1, padding=0)
        self.bn1_4 = nn.BatchNorm1d(n_feature_maps)

        # BLOCK2
        self.conv2_1  = nn.Conv1d(n_feature_maps, 2*n_feature_maps, kernel_size=7, stride=1, padding=3)
        self.bn2_1 = nn.BatchNorm1d(2*n_feature_maps)

        self.conv2_2  = nn.Conv1d(2*n_feature_maps, 2*n_feature_maps, kernel_size=5, stride=1, padding=2)
        self.bn2_2 = nn.BatchNorm1d(2*n_feature_maps)

        self.conv2_3  = nn.Conv1d(2*n_feature_maps, 2*n_feature_maps, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm1d(2*n_feature_maps)

        self.shortcut2 = nn.Conv1d(n_feature_maps, 2*n_feature_maps, kernel_size=1, stride=1, padding=0)
        self.bn2_4 = nn.BatchNorm1d(2*n_feature_maps)

        # BLOCK3
        self.conv3_1  = nn.Conv1d(2*n_feature_maps, 2*n_feature_maps, kernel_size=7, stride=1, padding=3)
        self.bn3_1 = nn.BatchNorm1d(2*n_feature_maps)

        self.conv3_2  = nn.Conv1d(2*n_feature_maps, 2*n_feature_maps, kernel_size=5, stride=1, padding=2)
        self.bn3_2 = nn.BatchNorm1d(2*n_feature_maps)

        self.conv3_3  = nn.Conv1d(2*n_feature_maps, 2*n_feature_maps, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm1d(2*n_feature_maps)

        self.shortcut3 = nn.Conv1d(2*n_feature_maps, 2*n_feature_maps, kernel_size=1, stride=1, padding=0)
        self.bn3_4 = nn.BatchNorm1d(2*n_feature_maps)

        # FINAL
        self.gap = nn.AvgPool1d(kernel_size=D)
        self.fc = nn.Linear(2*n_feature_maps, K)

    def forward(self, x):
        x = x.reshape(-1, 1, D)

        # BLOCK1
        out = self.conv1_1(x)
        out = F.relu(self.bn1_1(out))

        out = self.conv1_2(out)
        out = F.relu(self.bn1_2(out))

        out = self.conv1_3(out)
        out = F.relu(self.bn1_3(out))

        shortcut1 = self.shortcut1(x)
        shortcut1 = self.bn1_4(shortcut1)

        out += shortcut1
        out1 = F.relu(out)

        # BLOCK2
        out = self.conv2_1(out1)
        out = F.relu(self.bn2_1(out))

        out = self.conv2_2(out)
        out = F.relu(self.bn2_2(out))

        out = self.conv2_3(out)
        out = F.relu(self.bn2_3(out))

        shortcut2 = self.shortcut2(out1)
        shortcut2 = self.bn2_4(shortcut2)

        out += shortcut2
        out2 = F.relu(out)

        # BLOCK3
        out = self.conv3_1(out2)
        out = F.relu(self.bn3_1(out))

        out = self.conv3_2(out)
        out = F.relu(self.bn3_2(out))

        out = self.conv3_3(out)
        out = F.relu(self.bn3_3(out))

        shortcut3 = self.shortcut3(out2)
        shortcut3 = self.bn3_4(shortcut3)

        out += shortcut3
        out3 = F.relu(out)

        # FINAL
        out = self.gap(out3).reshape(-1, 2*n_feature_maps)
        out = self.fc(out)

        return F.softmax(out, dim=1)

