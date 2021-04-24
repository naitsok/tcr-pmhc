import glob
import os

import numpy as np
import pandas as pd
from pathlib import Path


#####
# Getting amino acid sequences
#####

def oneHot(residue):
    """
    Converts string sequence to one-hot encoding
    Example usage:
    seq = "GSHSMRY"
    oneHot(seq)
    """
    
    mapping = dict(zip("ACDEFGHIKLMNPQRSTVWY", range(20)))
    if residue in "ACDEFGHIKLMNPQRSTVWY":
        return np.eye(20)[mapping[residue]]
    else:
        return np.zeros(20)
    
def reverseOneHot(encoding):
    """
    Converts one-hot encoded array back to string sequence
    """
    mapping = dict(zip(range(20),"ACDEFGHIKLMNPQRSTVWY"))
    seq=''
    for i in range(len(encoding)):
        if np.max(encoding[i])>0:
            seq+=mapping[np.argmax(encoding[i])]
    return seq

def extract_sequences(dataset_X):
    """
    Return DataFrame with MHC, peptide and TCR a/b sequences from
    one-hot encoded complex sequences in dataset X
    """
    complex_sequences = [reverseOneHot(arr[:, 0:20]) for arr in dataset_X]
    mhc_sequences = [seq[0:180] for seq in complex_sequences]
    pep_sequences = [seq[180:192] for seq in complex_sequences]
    tcrab_sequences = [seq[192:] for seq in complex_sequences]
    df_sequences = pd.DataFrame({"MHC":mhc_sequences, "peptide":pep_sequences,
                                 "tcra":tcrab_sequences})
    return df_sequences


#####
# Load data and convert to amino acid sequences
#####

def load_data():
    data_list = []
    target_list = []

    for fp in glob.glob("./data/train/*input.npz"):
        data = np.load(fp)["arr_0"]
        targets = np.load(fp.replace("input", "labels"))["arr_0"]
        
        data_list.append(data)
        target_list.append(targets)

    X_data = np.concatenate(data_list)
    Y_data = np.concatenate(target_list)
    print("X data shape ", X_data.shape)
    print("Y data shape ", Y_data.shape)

    return X_data, Y_data


def convert_to_sequences_save_csv():
    X_data, Y_data = load_data()    
    X_seqs = extract_sequences(X_data)
    X_seqs.to_csv("./data/train/X_seqs.csv")
    pd.DataFrame(Y_data).to_csv("./data/train/Y_data.csv")


#####
# Embedding using https://github.com/Rostlab/SeqVec
#####

def embed_list_and_save(seq_lst, amino_acid_length, batch_size, name):
    # lst - 1d list of amino acid sequences

    folder = "./data/train/" + name
    if (not os.path.exists(folder)):
        os.mkdir(folder)

    from allennlp.commands.elmo import ElmoEmbedder

    model_dir = Path('./data/seqvec')
    weights = model_dir / 'weights.hdf5'
    options = model_dir / 'options.json'
    embedder = ElmoEmbedder(options, weights, cuda_device=0)

    num_batches = int(len(seq_lst) / batch_size)

    for j in range(num_batches):
        print("Embedding batch " + str(j) + " out of " + str(num_batches))
        trunc_seq_lst = seq_lst[(j * batch_size):(j + 1) * batch_size]

        embedding = np.zeros((batch_size, 3, amino_acid_length, 1024))
        for i, seq in enumerate(trunc_seq_lst):
            # print("Embedding sequence " + str(i) + " out of " + str(batch_size))
            embedding[i, :, :len(list(seq)), :] = embedder.embed_sentence(list(seq))

        np.save("./data/train/" + name + "/embedding_batch_" + str(j), embedding)


def embed_all():
    X_data = pd.read_csv("./data/train/X_seqs.csv", index_col=0)
    print(X_data.shape)

    batch_size = 16

    embed_list_and_save(X_data["peptide"], 12, batch_size, "peptide")
    embed_list_and_save(X_data["tcra"], 224, batch_size, "tcra")
    embed_list_and_save(X_data["MHC"], 180, batch_size, "MHC")


if __name__ == '__main__':
    embed_all()
