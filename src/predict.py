import argparse
import glob, os, tempfile, zipfile

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score

# Needed to predict
from model import CatCNN

parser = argparse.ArgumentParser()
parser.add_argument('--input-zip')
args = parser.parse_args()
print(args)

# Load files
filenames = []
dfs = []
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(args.input_zip) as zip:
        files = [file for file in zip.infolist() if file.filename.endswith(".npz")]
        for file in files:
            zip.extract(file, tmpdir)
            
        # Load npz files
        data_list = []

        for fp in glob.glob(tmpdir + "/*input.npz"):
            data = np.load(fp)["arr_0"]

            data_list.append(data)
         
X_test = np.concatenate(data_list[:])
nsamples, nx, ny = X_test.shape
print("test set shape:", nsamples,nx,ny)

# embed data, takes long time
from embed_sequences import extract_sequences, embed_list_and_save
X_seqs = extract_sequences(X_test)
# embed_list_and_save(X_seqs["tcra"], 224, 1, "tcra_test")
# embed_list_and_save(X_seqs["peptide"], 12, 1, "peptide_test")

def get_test_batch():
    # batch_size = 16
    for i in range(10): #X_test.shape[0]): 
        print("Validation batch: ", i + 1)
        X_tcr_test = np.load("./data/train/tcra_test/embedding_batch_" + str(i) + ".npy")
        X_pep_test = np.load("./data/train/peptide_test/embedding_batch_" + str(i) + ".npy")
        '''for j in range(1, 4):
            data = np.load("../data/train/tcra/embedding_batch_" + str(i + j) + ".npy")
            print(i+j)
            X_tcr_test = np.concatenate((X_tcr_test, np.load("./data/train/tcra_test/embedding_batch_" + str(i + j) + ".npy")), axis=0)
            X_pep_test = np.concatenate((X_pep_test, np.load("./data/train/peptide_test/embedding_batch_" + str(i + j) + ".npy")), axis=0)'''
        yield [X_tcr_test, X_pep_test]

'''test_ds = []
for i in range(len(X_test)):
    test_ds.append([np.transpose(X_test[i])])'''

# bat_size = 64
# print("\nNOTE:\nSetting batch-size to", bat_size)
# test_ldr = torch.utils.data.DataLoader(test_ds,batch_size=bat_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)

def predict(net):
    net.eval()
    test_preds = []
    with torch.no_grad():
        for batch_idx, data in enumerate(get_test_batch()): ###
            # x_batch_val = data[0].float().detach()
            # x_batch_val = x_batch_val.to(device)
            X_tcr_batch_val = torch.tensor(data[0], dtype = torch.float).to(device)
            X_pep_batch_val = torch.tensor(data[1], dtype = torch.float).to(device)

            output = net(X_tcr_batch_val, X_pep_batch_val)
            preds = np.round(output.cpu().detach())
            test_preds += list(preds.data.numpy().flatten()) 
        
    return (test_preds)

    

# import trained model
model = CatCNN().to(device)
model.load_state_dict(torch.load("./model.pt"))
model.eval()

y_pred = predict(model)

# Write y_true, y_pred to disk
outname = "./submission/predictions_catcnn.csv"
print("\nSaving TEST set y_pred to", outname)
df_performance = pd.DataFrame({"ix": range(len(y_pred)), "prediction": y_pred},)
df_performance.to_csv(outname, index=False)

# print(open(outname, 'r').read())
