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
from sklearn.model_selection import KFold

D = 178
K = 5

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--resnet', required=False, default='resnet', type=str)
parser.add_argument('-f', '--fcn', required=False, default=None, type=str)
parser.add_argument('-e', '--encoder', required=False, default=None, type=str)
args = parser.parse_args()

resnet_model = args.resnet
fcn_model = args.fcn
encoder_model = args.encoder

save_dir = 'checkpoints/'

device = torch.device('cpu')
print('device: ', device)

# single-batch load dataset
x_data, y_data = load_dataset('classification_data.csv', device)
n_folds = 5
kf = KFold(n_splits = n_folds, shuffle = True, random_state = 2)
scores = []
idx = 1
for train_index, test_index in tqdm(kf.split(x_data)):
    # init model
    nets = []
    if resnet_model is not None:
        net = ResNet()
        cp_name = resnet_model + str(idx)
        load_path = os.path.join(save_dir, cp_name)
        net.load_state_dict(torch.load(load_path)['state_dict'])
        nets.append(net)
    if fcn_model is not None:
        net = FCN()
        cp_name = fcn_model + str(idx)
        load_path = os.path.join(save_dir, cp_name)
        net.load_state_dict(torch.load(load_path)['state_dict'])
        nets.append(net)
    if encoder_model is not None:
        net = Encoder()
        cp_name = encoder_model + str(idx)
        load_path = os.path.join(save_dir, cp_name)
        net.load_state_dict(torch.load(load_path)['state_dict'])
        nets.append(net)

    x_train_data = [x_data[idx] for idx in train_index]
    y_train_data = [y_data[idx] for idx in train_index]
    x_test_data = [x_data[idx] for idx in test_index]
    y_test_data = [y_data[idx] for idx in test_index]

    # ensemble predict
    y_preds = ensemble_predict(nets, x_test_data)
    y_preds = [x.tolist() for x in y_preds]
    y_test_data = [x.tolist() for x in y_test_data]
    score = report(y_test_data, y_preds)
    scores.append(score)
    load_path = None
    idx += 1

print('Mean Accuracy: %.3f%%' % (sum(scores)*100/float(len(scores))))
