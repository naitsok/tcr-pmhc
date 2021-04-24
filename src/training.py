import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score

# torch.cuda.empty_cache()

###############################
###    Load data            ###
###############################
"""
data_list = []
target_list = []

for fp in glob.glob("../data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    
    data_list.append(data)
    target_list.append(targets)

# Note:
# Choose your own training and val set based on data_list and target_list
# Here using the last partition as val set

X_train = np.concatenate(data_list[ :-1])
y_train = np.concatenate(target_list[:-1])
nsamples, nx, ny = X_train.shape
print("Training set shape:", nsamples,nx,ny)

X_val = np.concatenate(data_list[-1: ])
y_val = np.concatenate(target_list[-1: ])
nsamples, nx, ny = X_val.shape
print("val set shape:", nsamples,nx,ny)

p_neg = len(y_train[y_train == 1])/len(y_train)*100
print("Percent positive samples in train:", p_neg)

p_pos = len(y_val[y_val == 1])/len(y_val)*100
print("Percent positive samples in val:", p_pos)

# make the data set into one dataset that can go into dataloader
train_ds = []
for i in range(len(X_train)):
    train_ds.append([np.transpose(X_train[i]), y_train[i]])

val_ds = []
for i in range(len(X_val)):
    val_ds.append([np.transpose(X_val[i]), y_val[i]])
"""

X_seqs = pd.read_csv("./data/train/X_seqs.csv", index_col=0)
Y_data = pd.read_csv("./data/train/Y_data.csv", index_col=0)
print("X sequences " , X_seqs.shape, ". Y data ", Y_data.shape)

bat_size = 16
print("\nSetting batch-size to", bat_size)

def get_train_batch(Y_data):
    saved_batch_size = 16
    batch_size = 64
    step = int(batch_size / saved_batch_size)
    y_list = Y_data.iloc[:, 0].tolist()
    for i in range(0, int(len(y_list) / saved_batch_size), step): #val_set_after):
        print("Training batch: ", i + 1)
        X_tcr = np.load("./data/train/tcra/embedding_batch_" + str(i) + ".npy")
        X_pep = np.load("./data/train/peptide/embedding_batch_" + str(i) + ".npy")
        Y_data = np.array(y_list[i * saved_batch_size:(i + step) * saved_batch_size])
        for j in range(1, step):
            data = np.load("./data/train/tcra/embedding_batch_" + str(i + j) + ".npy")
            # print(i+j)
            X_tcr = np.concatenate((X_tcr, np.load("./data/train/tcra/embedding_batch_" + str(i + j) + ".npy")), axis=0)
            X_pep = np.concatenate((X_pep, np.load("./data/train/peptide/embedding_batch_" + str(i + j) + ".npy")), axis=0)
            
        yield [X_tcr, X_pep, Y_data]

def get_val_batch(Y_data):
    batch_size = 16
    y_list_val = Y_data.iloc[:, 0].tolist()
    for j in range(270, 290): # int(len(y_list) / batch_size)):
        print("Validation batch: ", j + 1)
        X_tcr_val = np.load("./data/train/tcra/embedding_batch_" + str(j) + ".npy")
        X_pep_val = np.load("./data/train/peptide/embedding_batch_" + str(j) + ".npy")
        Y_data_val = np.array(y_list_val[j * batch_size:(j + 1) * batch_size])
        yield [X_tcr_val, X_pep_val, Y_data_val]

# train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=bat_size, shuffle=True)
# val_ldr = torch.utils.data.DataLoader(val_ds,batch_size=bat_size, shuffle=True)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
# device.empty_cache()



###############################
###    Define network       ###
###############################

print("Initializing network")

# Hyperparameters
# input_size = 420
# num_classes = 1
learning_rate = 0.01

from model import CatCNN
    
# Initialize network
net = CatCNN().to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)



###############################
###         TRAIN           ###
###############################

print("Training")

num_epochs = 3

train_acc, train_loss = [], []
valid_acc, valid_loss = [], []
losses = []
val_losses = []

for epoch in range(num_epochs):
    cur_loss = 0
    val_loss = 0
    print("\nEpoch: ", epoch + 1)

    net.train()
    train_preds, train_targs = [], [] 
    train_length = 0
    for k, data in enumerate(get_train_batch(Y_data)):
        # X_batch =  data.float().detach().requires_grad_(True)
        X_tcr_batch = torch.tensor(data[0], dtype = torch.float).to(device)
        X_pep_batch = torch.tensor(data[1], dtype = torch.float).to(device)
        target_batch = torch.tensor(data[2], dtype = torch.float).unsqueeze(1).to(device)
        # target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1)
        # X_batch, target_batch = X_batch.to(device), target_batch.to(device)
        
        
        optimizer.zero_grad()
        # output = net(X_batch)
        output = net(X_tcr_batch, X_pep_batch)
        
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()
        
        preds = np.round(output.detach().cpu())
        train_targs += list(np.array(target_batch.cpu()))
        train_preds += list(preds.data.numpy().flatten())
        cur_loss += batch_loss.detach()
        train_length += 1

    losses.append(cur_loss / train_length)
        
    
    net.eval()
    ### Evaluate validation
    val_preds, val_targs = [], []
    with torch.no_grad():
        val_length = 0
        for k, data_val in enumerate(get_val_batch(Y_data)):
            # x_batch_val = data.float().detach()
            # y_batch_val = target.float().detach().unsqueeze(1)
            # x_batch_val, y_batch_val = x_batch_val.to(device), y_batch_val.to(device)
            X_tcr_batch_val = torch.tensor(data_val[0], dtype = torch.float).to(device)
            X_pep_batch_val = torch.tensor(data_val[1], dtype = torch.float).to(device)
            target_batch = torch.tensor(data_val[2], dtype = torch.float).unsqueeze(1).to(device)
            
            # output = net(X_batch)
            output = net(X_tcr_batch_val, X_pep_batch_val)
            
            val_batch_loss = criterion(output, target_batch)
            
            preds = np.round(output.cpu().detach())
            val_preds += list(preds.data.numpy().flatten()) 
            val_targs += list(np.array(target_batch.cpu()))
            val_loss += val_batch_loss.detach()
            val_length += 1
            
        val_losses.append(val_loss / val_length)
        
        
        train_acc_cur = accuracy_score(train_targs, train_preds)  
        valid_acc_cur = accuracy_score(val_targs, val_preds) 

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        from sklearn.metrics import matthews_corrcoef
        print("Training loss:", losses[-1].item(), "Validation loss:", val_losses[-1].item(), end = "\n")
        print("MCC Train:", matthews_corrcoef(train_targs, train_preds), "MCC val:", matthews_corrcoef(val_targs, val_preds))
        print()
        
print('\nFinished Training ...')

# Write model to disk for use in predict.py
print("Saving model to src/model.pt")
torch.save(net.state_dict(), "./model.pt")
