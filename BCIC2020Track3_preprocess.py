"""
Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST)
Code for reproducing results on BCI Competition 2020 Track #3: Imagined Speech Classification.
Currently under review for publication.
Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)
"""

import os
import numpy as np
import scipy
import multiprocessing as mp
import multiprocessing.dummy as dmp
from functools import partial
import h5py

SRC_FOLDER = './'
DATA_FOLDER = './Processed'
NAME = 'BCIC2020Track3' # BCI Competition 2020 - Track 3 Imagined Speech Classification
SUBJECTS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
CLASSES = ['hello', 'help-me', 'stop', 'thank-you', 'yes']

Electrodes = [
    'Fp1','Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 
    'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
    'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
    'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
    'PO3', 'POz', 'PO4', 'PO8'
]

Zones = {
    'Pre-frontal': ['AF7', 'Fp1', 'Fp2', 'AF8', 'AF3', 'AF4'],
    'Frontal': ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    'Pre-central': ['FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
    'Central': ['C1', 'C2', 'C3', 'Cz', 'C4', 'C5', 'C6'],
    'Post-central': ['CP1', 'CP2', 'CP3', 'CPz', 'CP4', 'CP5', 'CP6'],
    'Temporal': ['T7', 'T8', 'FT7', 'FT8', 'TP7', 'TP8', 'TP9', 'TP10', 'FT9', 'FT10'],
    'Parietal': ['P1', 'P2', 'P3', 'P4', 'Pz', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'PO9', 'PO10'],
    'Occipital': ['O1', 'O2', 'Oz', 'POz'],
}

def proc_one(SID):
    data_train = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/Training set/Data_Sample{SID}.mat')
    data_valid = scipy.io.loadmat(f'{SRC_FOLDER}/{NAME}/Validation set/Data_Sample{SID}.mat')
    x_t, y_t = np.asarray(data_train['epo_train']['x'])[0][0], np.asarray(data_train['epo_train']['y'])[0][0].argmax(0)
    x_v, y_v = np.asarray(data_valid['epo_validation']['x'])[0][0], np.asarray(data_valid['epo_validation']['y'])[0][0].argmax(0)
    x_t = np.transpose(x_t, (2, 1, 0)).astype(np.float32)
    x_v = np.transpose(x_v, (2, 1, 0)).astype(np.float32)
    x, y = np.concatenate((x_t, x_v), axis=0), np.concatenate((y_t, y_v), axis=0).astype(np.uint8)
    x = np.pad(x, ((0, 0), (0, 0), (0, 5)), 'edge')
    print(SID, x.shape, y.shape)
    return SID, x, y

def proc_all():
    with mp.Pool(len(SUBJECTS)) as pool:
        with h5py.File(f'{DATA_FOLDER}/{NAME}.h5', 'w') as f:
            for SID, X, Y in pool.map(proc_one, SUBJECTS):
                f.create_dataset(f'{SID}/X', data=X)
                f.create_dataset(f'{SID}/Y', data=Y)
                print(SID, X.shape, Y.shape, np.unique(Y, return_counts=True))
            
if __name__ == '__main__':
    os.makedirs(DATA_FOLDER, exist_ok=True)
    proc_all()