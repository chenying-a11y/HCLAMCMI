import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix


def Gauss_M(adj_matrix, N):
    GM = np.zeros((N, N))
    rm = N * 1. / sum(sum(adj_matrix * adj_matrix))
    for i in range(N):
        for j in range(N):
            GM[i][j] = e ** (-rm * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GM


def Gauss_C(adj_matrix, M):
    GD = np.zeros((M, M))
    T = adj_matrix.transpose()
    rd = M * 1. / sum(sum(T * T))
    for i in range(M):
        for j in range(M):
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD


def prepare_data(opt):
    data = {}

    circ_sim = pd.read_csv('data/circrna_levenshtein_similarity.csv', header=None).values
    mirna_sim = pd.read_csv('data/mirna_levenshtein_similarity.csv', header=None).values
    assoc_matrix = pd.read_csv('data/adjacency_matrix.csv', header=None).values

    assoc_tensor = t.FloatTensor(assoc_matrix)
    data['assoc_matrix_raw'] = assoc_tensor
    data['assoc_matrix_truth'] = assoc_tensor

    pos_indices, neg_indices = [], []
    for i in range(assoc_tensor.size(0)):
        for j in range(assoc_tensor.size(1)):
            if assoc_tensor[i, j] >= 1:
                pos_indices.append([i, j])
            else:
                neg_indices.append([i, j])

    np.random.seed(0)
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    pos_tensor = t.LongTensor(pos_indices)
    neg_tensor = t.LongTensor(neg_indices)

    pos_splits = pos_tensor.split(int(len(pos_tensor) / 10), dim=0)
    neg_splits = neg_tensor.split(int(len(neg_tensor) / 10), dim=0)

    train_pos = t.cat(pos_splits[:9])
    train_neg = t.cat(neg_splits[:9])

    fold_pos = train_pos.split(int(len(train_pos) / opt.validation), dim=0)
    fold_neg = train_neg.split(int(len(train_neg) / opt.validation), dim=0)

    data['folds'] = []
    for fold_id in range(opt.validation):
        train_idx = [i for i in range(opt.validation) if i != fold_id]
        train_pos_fold = t.cat([fold_pos[i] for i in train_idx])
        train_neg_fold = t.cat([fold_neg[i] for i in train_idx])
        test_pos_fold = fold_pos[fold_id]
        test_neg_fold = fold_neg[fold_id]
        data['folds'].append({
            'train': [train_pos_fold, train_neg_fold],
            'test': [test_pos_fold, test_neg_fold]
        })

    data['independent_test'] = [{
        'train': [train_pos, train_neg],
        'test': [pos_splits[-2], neg_splits[-2]]}]

    circ_gauss = Gauss_C(assoc_matrix, assoc_matrix.shape[1])
    mirna_gauss = Gauss_M(assoc_matrix, assoc_matrix.shape[0])

    circ_combined = np.where(circ_sim == 0, circ_gauss, (circ_sim + circ_gauss) / 2)
    mirna_combined = np.where(mirna_sim == 0, mirna_gauss, (mirna_sim + mirna_gauss) / 2)

    data['circ_graph'] = t.from_numpy(circ_combined)
    data['mirna_graph'] = t.from_numpy(mirna_combined)

    return data

class Dataset(object):
    def __init__(self, opt, data):
        self.data = data
        self.num_folds = opt.validation

    def __getitem__(self, index):
        return (
            self.data['circ_graph'],
            self.data['mirna_graph'],
            self.data['folds'][index]['train'],
            self.data['folds'][index]['test'],
            self.data['assoc_matrix_raw'],
            self.data['assoc_matrix_truth'],
            self.data['independent_test'][0]['train'],
            self.data['independent_test'][0]['test']
        )

    def __len__(self):
        return self.num_folds
