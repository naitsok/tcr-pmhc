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

# ML architecture


class CatCNN(nn.Module):
    
    def __init__(self):
        super(CatCNN, self).__init__()

        self.num_classes = 1

        # Layers for TCR
        self.convTCR1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 5), stride=(2, 4), padding=(1, 3))
        torch.nn.init.kaiming_uniform_(self.convTCR1.weight)
        self.poolTCR = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bnTCR1 = nn.BatchNorm2d(6)
        self.convTCR2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 5), stride=(2, 4), padding=(1, 3))
        torch.nn.init.kaiming_uniform_(self.convTCR2.weight)
        self.bnTCR2 = nn.BatchNorm2d(12)
        # self.convTCR3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 5), stride=(2, 4), padding=(1, 3))

        # Layers for peptides
        self.convPep1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(1, 5), stride=(1, 4), padding=(0, 3))
        torch.nn.init.kaiming_uniform_(self.convPep1.weight)
        self.poolPep = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.bnPep1 = nn.BatchNorm2d(6)
        self.convPep2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3, 5), stride=(2, 4), padding=(1, 3))
        torch.nn.init.kaiming_uniform_(self.convPep2.weight)
        self.bnPep2 = nn.BatchNorm2d(12)

        # After combining TCR and peptides
        self.fc1 = nn.Linear(3840, 200)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, self.num_classes)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, X_tcr, X_pep):
        # TCR part
        X_tcr = self.poolTCR(F.relu(self.convTCR1(X_tcr)))
        X_tcr = self.bnTCR1(X_tcr)

        X_tcr = self.poolTCR(F.relu(self.convTCR2(X_tcr)))
        X_tcr = self.bnTCR2(X_tcr)

        # Peptide part
        X_pep = self.poolPep(F.relu(self.convPep1(X_pep)))
        X_pep = self.bnPep1(X_pep)

        X_pep = self.poolPep(F.relu(self.convPep2(X_pep)))
        X_pep = self.bnPep2(X_pep)

        # conbine TCR and Peptide
        X_tcr = X_tcr.view(X_tcr.size(0), -1)
        X_pep = X_pep.view(X_pep.size(0), -1)
        # print(X_tcr.shape)
        # print(X_pep.shape)
        x = torch.cat((X_tcr, X_pep), 1)
        # print(x.shape)

        # continue
        x = self.bn1(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        return x



class Net(nn.Module):
    num_classes = 1
    def __init__(self,  num_classes):
        super(Net, self).__init__()       
        self.conv1 = nn.Conv1d(in_channels=54, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1_bn = nn.BatchNorm1d(100)
        
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm1d(100)
        
        self.fc1 = nn.Linear(2600, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
    def forward(self, x):    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1_bn(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv2_bn(x)
        
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        
        return x
