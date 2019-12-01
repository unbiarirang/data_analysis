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
save_path = os.path.join(save_dir, model_name)
load_path = None
if checkpoint_name is not None:
    load_path = os.path.join(save_dir, checkpoint_name)


# init model
if model_name == 'resnet':
    net = ResNet()
    net.apply(weights_init_glorot_uniform)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=0.0001)
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1}
elif model_name == 'fcn':
    net = FCN()
    net.apply(weights_init_glorot_uniform)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=0.0001)
    params = {'batch_size': 16,
              'shuffle': True,
              'num_workers': 1}
elif model_name == 'encoder':
    net = Encoder()
    net.apply(weights_init_glorot_uniform)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    params = {'batch_size': 12,
              'shuffle': True,
              'num_workers': 1}

# use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
net.to(device)

## batch load dataset
#train_loader, test_loader = load_dataset_batch('classification_data.csv', params, device)
#
## batch train
#if model_name == 'resnet' or model_name == 'fcn':
#    train_scheduler_batch(net, epochs, scheduler, optimizer, criterion, train_loader, test_loader, save_path, load_path)
#elif model_name == 'encoder':
#    train_batch(net, epochs, optimizer, criterion, train_loader, test_loader, save_path, load_path)

# single-batch load dataset
x_data, y_data = load_dataset('classification_data.csv', device)

# 5-folds validation
n_folds = 5
kf = KFold(n_splits = n_folds, shuffle = True, random_state = 2)
scores = []
for train_index, test_index in tqdm(kf.split(x_data)):
    x_train_data = [x_data[idx] for idx in train_index]
    y_train_data = [y_data[idx] for idx in train_index]
    x_test_data = [x_data[idx] for idx in test_index]
    y_test_data = [y_data[idx] for idx in test_index]

    # single-batch train
    if model_name == 'resnet' or model_name == 'fcn':
        # train with scheduler
        train_scheduler(net, epochs, scheduler, optimizer, criterion, x_train_data, y_train_data, x_test_data, y_test_data, save_path, load_path)
    elif model_name == 'encoder':
        train(net, epochs, optimizer, criterion, x_train_data, y_train_data, x_test_data, y_test_data, save_path, load_path)

    # predict
    y_preds = predict(net, x_test_data)
    score = report(y_test_data, y_preds)
    scores.append(score)

print('Mean Accuracy: %.3f%%' % (sum(scores)*100/float(len(scores))))
