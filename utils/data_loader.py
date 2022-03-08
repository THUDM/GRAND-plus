import sys

import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
from utils.make_dataset import get_dataset, get_train_val_test_split
import os
from sklearn import metrics


def load_data(dataset_str='cora', split_seed=0, renormalize=False):
    """Load data."""
    if  os.path.exists("dataset/"):
        path = "dataset/"
    else:
        path = "../dataset/"
    if dataset_str == 'aminer':
        adj = pkl.load(open(path + "{}/{}.adj.sp.pkl".format(dataset_str, dataset_str), "rb"))
        features = pkl.load(
            open(path + "{}/{}.features.pkl".format(dataset_str, dataset_str), "rb"))
        labels = pkl.load(
            open(path + "{}/{}.labels.pkl".format(dataset_str, dataset_str), "rb"))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        features = col_normalize(features)

    elif dataset_str in ['ms_academic_cs', 'ms_academic_phy', 'amazon_electronics_photo', 'amazon_electronics_computers', 'cora_full']:
        datapath = path + dataset_str + '.npz'
        adj, features, labels = get_dataset(
            dataset_str, datapath, True, train_examples_per_class=20, val_examples_per_class=30)
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        features = features.todense()
    
    elif dataset_str in ['reddit']:
        adj = sp.load_npz(path + '{}/{}_adj.npz'.format(dataset_str, dataset_str))
        features = np.load(path + '{}/{}_feat.npy'.format(dataset_str, dataset_str))
        labels = np.load(path + '{}/{}_labels.npy'.format(dataset_str, dataset_str)) 
        print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
        idx_unlabel = np.concatenate((idx_val, idx_test))
        print(dataset_str, features.shape)
    
    elif dataset_str in ['Amazon2M']:
        adj = sp.load_npz(path + '{}/{}_adj.npz'.format(dataset_str, dataset_str))
        features = np.load(path + '{}/{}_feat.npy'.format(dataset_str, dataset_str))
        labels = np.load(path + '{}/{}_labels.npy'.format(dataset_str, dataset_str))
        print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20* class_num, val_size=30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    
    elif dataset_str in ['mag_scholar_c', 'mag_scholar_f']:
        data_set = np.load(path + dataset_str + '.npz') 
        adj_data = data_set['adj_matrix.data']
        adj_indices = data_set['adj_matrix.indices']
        adj_indptr = data_set['adj_matrix.indptr']
        adj_shape = data_set['adj_matrix.shape']

        feat_data = data_set['attr_matrix.data']
        feat_indices = data_set['attr_matrix.indices']
        feat_indptr = data_set['attr_matrix.indptr']
        feat_shape = data_set['attr_matrix.shape']
        labels_num = data_set['labels']
        features = sp.csr_matrix((feat_data, feat_indices, feat_indptr), shape=feat_shape)
        adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape)
        random_state = np.random.RandomState(split_seed)
        label_count = labels_num.max() + 1
        labels = np.eye(label_count)[labels_num]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))

    elif dataset_str in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(path + "citation/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            path + "citation/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        # normalize
        features = normalize(features)
        features = features.todense()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_train = np.arange(len(y))
        idx_val = np.arange(len(y), len(y)+500)
        idx_test = np.asarray(test_idx_range.tolist())
        idx_unlabel = np.arange(len(y), labels.shape[0])
    else:
        raise NotImplementedError

    if renormalize:
        adj = adj + sp.eye(adj.shape[0])
        D1 = np.array(adj.sum(axis=1))**(-0.5)
        D2 = np.array(adj.sum(axis=0))**(-0.5)
        D1 = sp.diags(D1[:, 0], format='csr')
        D2 = sp.diags(D2[0, :], format='csr')

        A = adj.dot(D1)
        A = D2.dot(A)
        adj = A

    return adj, features, labels, idx_train, idx_val, idx_test, idx_unlabel

def totensor(features, labels):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.argmax(labels, -1))

    return features, labels 


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def col_normalize(mx):
    """Column-normalize sparse matrix"""
    scaler = StandardScaler()

    mx = scaler.fit_transform(mx)

    return mx


