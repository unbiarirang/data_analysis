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

# constants
D = 178
K = 5

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', required=False, default=1, type=int)
parser.add_argument('-m', '--model', required=True, type=str)
parser.add_argument('-c', '--checkpoint', required=False, default=None, type=str)
args = parser.parse_args()
epochs = args.epochs
model_name = args.model
checkpoint_name = args.checkpoint

save_dir = 'checkpoints/'
load_path = None
if checkpoint_name is not None:
    load_path = os.path.join(save_dir, checkpoint_name)


# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# load dataset
x_data, y_data = load_dataset('classification_data.csv', device)

# 5-folds validation
n_folds = 5
kf = KFold(n_splits = n_folds, shuffle = True, random_state = 2)
scores = []
idx = 1
for train_index, test_index in tqdm(kf.split(x_data)):
    # init model
    cp_name = model_name + str(idx)
    save_path = os.path.join(save_dir, cp_name)
    if 'resnet' in cp_name:
        net = ResNet()
        net.apply(weights_init_glorot_uniform)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=0.0001)
        params = {'batch_size': 16,
                  'shuffle': True,
                  'num_workers': 1}
    elif 'fcn' in cp_name:
        net = FCN()
        net.apply(weights_init_glorot_uniform)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=0.0001)
        params = {'batch_size': 16,
                  'shuffle': True,
                  'num_workers': 1}
    elif 'encoder' in cp_name:
        net = Encoder()
        net.apply(weights_init_glorot_uniform)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.00001)
        params = {'batch_size': 12,
                  'shuffle': True,
                  'num_workers': 1}

    net.to(device)

    x_train_data = [x_data[idx] for idx in train_index]
    y_train_data = [y_data[idx] for idx in train_index]
    x_test_data = [x_data[idx] for idx in test_index]
    y_test_data = [y_data[idx] for idx in test_index]

    # train
    if 'resnet' in cp_name or 'fcn' in cp_name:
        # train with scheduler
        train_scheduler(net, epochs, scheduler, optimizer, criterion, x_train_data, y_train_data, x_test_data, y_test_data, save_path, load_path)
    elif 'encoder' in cp_name:
        train(net, epochs, optimizer, criterion, x_train_data, y_train_data, x_test_data, y_test_data, save_path, load_path)

    # predict
    load_path = os.path.join(save_dir, cp_name)
    _, _ = load_checkpoint(net, optimizer, load_path)
    y_preds = predict(net, x_test_data)
    y_preds = [x.tolist() for x in y_preds]
    y_test_data = [x.tolist() for x in y_test_data]
    score = report(y_test_data, y_preds)
    scores.append(score)
    load_path = None
    idx += 1

print('Mean Accuracy: %.3f%%' % (sum(scores)*100/float(len(scores))))
