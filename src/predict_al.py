# TEST

import argparse
import glob, os, tempfile, zipfile

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score


def extract_energy(dataset):
    energy = np.array([arr[0, 27:] for arr in dataset])
    df = pd.DataFrame()
    for i in range(27):
        df['V{}'.format(i)] = energy[:,i]
    return df


parser = argparse.ArgumentParser()
parser.add_argument('--input-zip')
args = parser.parse_args()
print(args)

#tmpdir = "/Users/tsukanov/Documents/biohacaton/tcr-pmhc/submission/"
# Load files
filenames = []
dfs = []
with tempfile.TemporaryDirectory() as tmpdir:
#    with zipfile.ZipFile("/Users/tsukanov/Documents/biohacaton/tcr-pmhc/submission/input.zip") as zip:
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
prepared_data = extract_energy(X_test)


# import trained model
#model = lgb.Booster(model_file="/Users/tsukanov/Documents/biohacaton/tcr-pmhc/src/model.txt")
model = lgb.Booster(model_file="src/model.txt")
proba_model = model.predict(prepared_data)
model_results = np.array([1 if i > 0.4 else 0 for i in proba_model])

# Write y_true, y_pred to disk
outname = "predictions.csv"
print("\nSaving TEST set y_pred to", outname)
df_performance = pd.DataFrame({"ix": range(len(model_results)), "prediction": model_results},)
df_performance.to_csv(outname, index=False)

print(open(outname, 'r').read())
