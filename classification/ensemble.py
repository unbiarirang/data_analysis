import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import argparse
from tqdm import tqdm
from utils import *
from resnet import ResNet
from fcn import FCN
from encoder import Encoder

D = 178
K = 5

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resnet', required=True, default='resnet', type=str)
parser.add_argument('-f', '--fcn', required=False, default=None, type=str)
parser.add_argument('-e', '--encoder', required=False, default=None, type=str)
args = parser.parse_args()

resnet_model = args.resnet
fcn_model = args.fcn
encoder_model = args.encoder

save_dir = 'checkpoints/'

# load models
nets = []
if resnet_model is not None:
    net = ResNet()
    load_path = os.path.join(save_dir, resnet_model)
    net.load_state_dict(torch.load(load_path)['state_dict'])
    nets.append(net)
if fcn_model is not None:
    net = FCN()
    load_path = os.path.join(save_dir, fcn_model)
    net.load_state_dict(torch.load(load_path)['state_dict'])
    nets.append(net)
if encoder_model is not None:
    net = Encoder()
    load_path = os.path.join(save_dir, encoder_model)
    net.load_state_dict(torch.load(load_path)['state_dict'])
    nets.append(net)

print('ensemble {} models {}'.format(len(nets), nets))

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
net.to(device)

# single-batch load dataset
x_train_data, y_train_data, x_test_data, y_test_data = load_dataset('classification_data.csv', device)

# ensemble predict
y_preds = ensemble_predict(nets, x_test_data)
report(y_test_data, y_preds)
