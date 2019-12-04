import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

D = 178
K = 5

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # BLOCK1
        self.conv1    = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.in1      = nn.InstanceNorm1d(128)
        self.prelu1   = nn.PReLU(128)
        self.dropout1 = nn.Dropout(0.2)
        self.maxpool1 = nn.MaxPool1d(2)

        # BLOCK2
        self.conv2    = nn.Conv1d(128, 256, kernel_size=11, padding=5)
        self.in2      = nn.InstanceNorm1d(256)
        self.prelu2   = nn.PReLU(256)
        self.dropout2 = nn.Dropout(0.2)
        self.maxpool2 = nn.MaxPool1d(2)

        # BLOCK3
        self.conv3    = nn.Conv1d(256, 512, kernel_size=21, padding=10)
        self.in3      = nn.InstanceNorm1d(512)
        self.prelu3   = nn.PReLU(512)
        self.dropout3 = nn.Dropout(0.2)

        # FINAL
        self.fc1  = nn.Linear(44, 256)
        self.in4   = nn.InstanceNorm1d(256)
        self.fc2  = nn.Linear(256 * 256, K)

    def forward(self, x):
        x = x.reshape(-1, 1, D)

        # BLOCK1
        out = self.conv1(x)
        out = self.in1(out)
        out = self.prelu1(out)
        out = self.dropout1(out)
        out = self.maxpool1(out)

        # BLOCK2
        out = self.conv2(out)
        out = self.in2(out)
        out = self.prelu2(out)
        out = self.dropout2(out)
        out = self.maxpool2(out)

        # BLOCK3
        out = self.conv3(out)
        out = self.in3(out)
        out = self.prelu3(out)
        out = self.dropout3(out)

        # ATTENTION
        attention_data = out[:,:256,:]
        attention_softmax = out[:,256:,:]
        attention_softmax = F.softmax(attention_softmax, dim=1)
        mul = attention_softmax * attention_data

        # FINAL
        out = torch.sigmoid(self.fc1(mul))
#        out = self.in4(out.reshape(1, -1))
        out = self.in4(out.reshape(-1, 256*256))
        out = self.fc2(out)

        return F.softmax(out, dim=1)

