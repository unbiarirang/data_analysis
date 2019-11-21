import csv
import torch
from torch.utils import data
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

D = 178

def read_csv(input_file):
    with open(input_file, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=',')
        lines = []
        for line in reader:
            lines.append(line)
    return lines[1:] # remove the first line

# multi-class data for neural network (one more dimension)
def load_data_nn(lines):
    x_train_data = []
    y_train_data = []
    for line in lines[:len(lines)//2]:
        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_train_data.append([x])
        y_train_data.append([int(line[-1]) - 1]) # index

    x_test_data= []
    y_test_data = []
    for line in lines[len(lines)//2:]:
        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_test_data.append([x])
        y_test_data.append([int(line[-1]) - 1]) # index

    return x_train_data, y_train_data, x_test_data, y_test_data

# multi-class data
def load_data(lines):
    x_train_data = []
    y_train_data = []
    for line in lines[:len(lines)//2]:
        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_train_data.append(x)
        y_train_data.append(int(line[-1]) - 1) # index

    x_test_data= []
    y_test_data = []
    for line in lines[len(lines)//2:]:
        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_test_data.append(x)
        y_test_data.append(int(line[-1]) - 1) # index

    return x_train_data, y_train_data, x_test_data, y_test_data

# multi-class data
def load_abnormal_data(lines):
    x_train_data = []
    y_train_data = []
    for line in lines[:len(lines)//2]:
        y = int(line[-1]) -1
        if y == 0: continue         # only collect abnormal data
        y_train_data.append(y)      # index

        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_train_data.append(x)

    x_test_data= []
    y_test_data = []
    for line in lines[len(lines)//2:]:
        y = int(line[-1]) -1
        if y == 0: continue         # only collect abnormal data
        y_test_data.append(y)      # index

        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_test_data.append(x)

    return x_train_data, y_train_data, x_test_data, y_test_data

def load_binary_data(lines, base=0):
    x_train_data = []
    y_train_data = []
    for line in lines[:len(lines)//2]:
        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_train_data.append(x)
        y = int(line[-1]) - 1       # index
        y = 0 if y == base else 1   # binary classification
        y_train_data.append(y)      # index

    x_test_data= []
    y_test_data = []
    for line in lines[len(lines)//2:]:
        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_test_data.append(x)
        y = int(line[-1]) - 1 # index
        y = 0 if y == 0 else 1 # binary classification
        y_test_data.append(y) # index

    return x_train_data, y_train_data, x_test_data, y_test_data

def load_binary_abnormal_data(lines, base):
    x_train_data = []
    y_train_data = []
    for line in lines[:len(lines)//2]:
        y = int(line[-1]) - 1       # index
        if y == 0: continue         # only collect abnormal data
        y = 0 if y == base else 1   # binary classification
        y_train_data.append(y)      # index

        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_train_data.append(x)

    x_test_data= []
    y_test_data = []
    for line in lines[len(lines)//2:]:
        y = int(line[-1]) - 1       # index
        if y == 0: continue         # only collect abnormal data
        y = 0 if y == 0 else 1 # binary classification
        y_test_data.append(y) # index

        x = line[1:-1] # remove the first column and the last column
        x = [int(i) for i in x]
        x_test_data.append(x)

    return x_train_data, y_train_data, x_test_data, y_test_data

class Dataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def load_dataset(file_path):
    lines = read_csv(file_path)
    x_train_data, y_train_data, x_test_data, y_test_data = load_data_nn(lines)
    x_train_data, x_test_data = feature_scaler(x_train_data, x_test_data)
#    x_train_data, y_train_data, x_test_data, y_test_data = x_train_data[:50], y_train_data[:50], x_test_data[:50], y_test_data[:50]
    x_train_data, y_train_data, x_test_data, y_test_data = \
        torch.FloatTensor(x_train_data), torch.LongTensor(y_train_data), \
        torch.FloatTensor(x_test_data), torch.LongTensor(y_test_data)
    return x_train_data, y_train_data, x_test_data, y_test_data

def load_dataset_batch(file_path, params):
    lines = read_csv(file_path)
    x_train_data, y_train_data, x_test_data, y_test_data = load_data_nn(lines)
    x_train_data, x_test_data = feature_scaler(x_train_data, x_test_data)
#    x_train_data, y_train_data, x_test_data, y_test_data = x_train_data[:50], y_train_data[:50], x_test_data[:50], y_test_data[:50]
    x_train_data, y_train_data, x_test_data, y_test_data = \
        torch.FloatTensor(x_train_data), torch.LongTensor(y_train_data), \
        torch.FloatTensor(x_test_data), torch.LongTensor(y_test_data)
    train_set = Dataset(x_train_data, y_train_data)
    train_loader = data.DataLoader(train_set, **params)
    test_set = Dataset(x_test_data, y_test_data)
    test_loader = data.DataLoader(test_set, **params)
    return train_loader, test_loader

#def feature_scaler(X, X_test):
#    mean = np.mean(X, axis=0)
#    std = np.std(X, axis=0)
#    mean_mat = np.tile(mean, (len(X), 1))
#    std_mat = np.tile(std, (len(X), 1))
#    X = (X - mean_mat) / std_mat
#    X_test = (X_test - mean_mat) / std_mat
#    return X, X_test

def feature_scaler(X, X_test):
    X = np.array(X).reshape(-1, D)
    X_test = np.array(X_test).reshape(-1, D)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X), scaler.transform(X_test)

def feature_reduction(x_train_data, x_test_data, newD):
    # PCA - feature reduction
    pca = PCA(n_components=newD, svd_solver='randomized', whiten=True).fit(x_train_data)
    return pca.transform(x_train_data), pca.transform(x_test_data)

def report(y_test_data, y_preds):
    print(confusion_matrix(y_test_data, y_preds))
    print(classification_report(y_test_data, y_preds))
    print('accuracy: ', accuracy_score(y_test_data, y_preds))
#    print('precision: ', precision_score(y_test_data, y_preds, average='macro'))

def train(net, epochs, optimizer, criterion, x_train_data, y_train_data, x_test_data, y_test_data, save_path, load_path):
    start_epoch = 1
    best_loss = 1000000.0

    # load checkpoint
    if load_path is not None:
        start_epoch, best_loss = load_checkpoint(net, optimizer, load_path)
        start_epoch += 1 # next epoch of the checkpoint

    for epoch in tqdm(range(start_epoch, epochs+1)):
        for i, (x, y) in enumerate(zip(x_train_data, y_train_data)):
            net.train()
            optimizer.zero_grad()
            # Forward pass
            y_pred = net(x)
            # Compute Loss
            loss = criterion(y_pred, y)
            # Backward pass
            loss.backward()
            optimizer.step()
        print('=> Ep:', epoch, 'loss:', loss.item())

        # eval every 5 epochs
        if epoch % 5 == 0:
            eval_loss = evaluate(net, criterion, x_test_data, y_test_data)
            print('eval loss:', eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(net, optimizer, epoch, eval_loss, save_path)

def train_scheduler(net, epochs, scheduler, optimizer, criterion, x_train_data, y_train_data, x_test_data, y_test_data, save_path, load_path):
    start_epoch = 1
    best_loss = 1000000.0

    # load checkpoint
    if load_path is not None:
        start_epoch, best_loss = load_checkpoint(net, optimizer, load_path)
        start_epoch += 1 # next epoch of the checkpoint

    for epoch in tqdm(range(start_epoch, epochs+1)):
        for _, (x, y) in tqdm(enumerate(zip(x_train_data, y_train_data))):
            net.train()
            optimizer.zero_grad()
            # Forward pass
            y_pred = net(x)
            # Compute Loss
            loss = criterion(y_pred, y)
            # Backward pass
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
        print('=> Ep:', epoch, 'loss:', loss.item(), \
          'lr:', [group['lr'] for group in optimizer.param_groups][0])

        # eval every 10 epochs
        if epoch % 10 == 0:
            eval_loss = evaluate(net, criterion, x_test_data, y_test_data)
            print('eval loss:', eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(net, optimizer, epoch, eval_loss, save_path)

def train_batch(net, epochs, optimizer, criterion, train_loader, test_loader, save_path, load_path):
    start_epoch = 1
    best_loss = 1000000.0

    # load checkpoint
    if load_path is not None:
        start_epoch, best_loss = load_checkpoint(net, optimizer, load_path)
        start_epoch += 1 # next epoch of the checkpoint

    for epoch in tqdm(range(start_epoch, epochs+1)):
        for _, (x, y) in tqdm(enumerate(train_loader)):
            net.train()
            optimizer.zero_grad()
            # Forward pass
            y_pred = net(x)
            # Compute Loss
            loss = criterion(y_pred, y.reshape(-1))
            # Backward pass
            loss.backward()
            optimizer.step()
        print('=> Ep:', epoch, 'loss:', loss.item())

        # eval every 10 epochs
        if epoch % 10 == 0:
            eval_loss = evaluate_batch(net, criterion, test_loader)
            print('eval loss:', eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(net, optimizer, epoch, eval_loss, save_path)

def train_scheduler_batch(net, epochs, scheduler, optimizer, criterion, train_loader, test_loader, save_path, load_path):
    start_epoch = 1
    best_loss = 1000000.0

    # load checkpoint
    if load_path is not None:
        start_epoch, best_loss = load_checkpoint(net, optimizer, load_path)
        start_epoch += 1 # next epoch of the checkpoint

    for epoch in tqdm(range(start_epoch, epochs+1)):
        for _, (x, y) in tqdm(enumerate(train_loader)):
            net.train()
            optimizer.zero_grad()
            # Forward pass
            y_pred = net(x)
            # Compute Loss
            loss = criterion(y_pred, y.reshape(-1))
            # Backward pass
            loss.backward()
            optimizer.step()
        scheduler.step(loss)
        print('=> Ep:', epoch, 'loss:', loss.item(), \
          'lr:', [group['lr'] for group in optimizer.param_groups][0])

        # eval every 10 epochs
        if epoch % 10 == 0:
            eval_loss = evaluate_batch(net, criterion, test_loader)
            print('eval loss:', eval_loss)
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(net, optimizer, epoch, eval_loss, save_path)

def predict(net, X):
    y_preds = []
    for x in tqdm(X):
        output = net(x)
        _, y_pred = torch.max(output.data, 1)
        y_preds.append(y_pred)
    return y_preds

def evaluate(net, criterion, X, Y):
    y_preds = []
    losses = []
    for (x, y) in tqdm(zip(X, Y)):
        output = net(x)
        loss = criterion(output, y).sum().item()
        losses.append(loss)
    return sum(losses) / len(losses)

def evaluate_batch(net, criterion, test_loader):
    y_preds = []
    losses = []
    for (x, y) in tqdm(test_loader):
        output = net(x)
        loss = criterion(output, y.reshape(-1)).sum().item()
        losses.append(loss)
    return sum(losses) / len(losses)

def weights_init_glorot_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)

def save_checkpoint(net, optimizer, epoch, eval_loss, save_path):
    torch.save({
            'epoch': epoch,
            'eval_loss': eval_loss,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, save_path)
    print('save checkpoint {} epoch {}'.format(save_path, epoch))

def load_checkpoint(net, optimizer, save_path):
    checkpoint = torch.load(save_path)
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('=> loaded checkpoint {} epoch {} best eval loss {}'.format(save_path, checkpoint['epoch'], checkpoint['eval_loss']))
    return checkpoint['epoch'], checkpoint['eval_loss']

