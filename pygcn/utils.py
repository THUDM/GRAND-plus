import sys

import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import torch
#import torch_sparse
import math
import torch.nn as nn
import torch.nn.functional as F
from gnn_bench_data.make_dataset import get_dataset, get_train_val_test_split
import os
from sklearn import metrics


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset_str='cora', split_seed=0, renormalize=False):
    """Load data."""
    if  os.path.exists("data/"):
        path = "data/"
    else:
        path = "../data/"
    if dataset_str == 'aminer':
        adj = pkl.load(open(path + "{}.adj.sp.pkl".format(dataset_str), "rb"))
        features = pkl.load(
            open(path + "{}.features.pkl".format(dataset_str), "rb"))
        labels = pkl.load(
            open(path + "{}.labels.pkl".format(dataset_str), "rb"))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)
        idx_unlabel = np.concatenate((idx_val, idx_test))
        features = col_normalize(features)
        # return adj, features, labels, idx_train, idx_val, idx_test, idx_unlabel

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
        adj = sp.load_npz(path + dataset_str + '_adj.npz')
        features = np.load(path + dataset_str + '_feat.npy')
        labels = np.load(path + dataset_str + '_labels.npy') 
        print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        idx_train, idx_val, idx_test = get_train_val_test_split(
            random_state, labels, train_examples_per_class=20, val_examples_per_class=30)    
        idx_unlabel = np.concatenate((idx_val, idx_test))
        print(dataset_str, features.shape)

    elif dataset_str in ['yelp','amazon']:
        adj = sp.load_npz(path + dataset_str + '_adj.npz')
        features = np.load(path + dataset_str + '_feat.npy')
        labels = np.load(path + dataset_str + '_labels.npy')
        print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20 * class_num, val_size= 30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    
    elif dataset_str in ['Amazon2M']:
        path = '/data/wenzheng/'
        adj = sp.load_npz(path + dataset_str + '_adj.npz')
        features = np.load(path + dataset_str + '_feat.npy')
        labels = np.load(path + dataset_str + '_labels.npy')
        print(labels.shape, list(np.sum(labels, axis=0)))
        random_state = np.random.RandomState(split_seed)
        class_num = labels.shape[1]
        idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_size=20* class_num, val_size=30 * class_num)
        idx_unlabel = np.concatenate((idx_val, idx_test))
    
    elif dataset_str in ['mag_scholar_c', 'mag_scholar_f']:
        path_mag = "/data/wenzheng/"
        if os.path.exists(path + dataset_str + '.npz'):
            data_set = np.load(path + dataset_str + '.npz') 
        else:
            data_set = np.load(path_mag + dataset_str + '.npz')
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
            with open(path + "ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            path + "ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
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

        # network_emb = pros(adj)
        # network_emb = 0

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]     # onehot

        idx_train = np.arange(len(y))
        idx_val = np.arange(len(y), len(y)+500)
        idx_test = np.asarray(test_idx_range.tolist())
        #features = features.todense()
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

def data_split(labels, split_seed, train_num = 20, val_num = 30):
    label_count = labels.shape[1]
    #labels = np.eye(label_count)[labels_num]
    random_state = np.random.RandomState(split_seed)
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels, train_examples_per_class=train_num, val_examples_per_class=val_num)
    idx_unlabel = np.concatenate((idx_val, idx_test))
    return idx_train, idx_val, idx_test, idx_unlabel



def D_adj_norm(adj, deg=None, sort_indices=True):
    """
    Normalize adj according to the method of rw normalization.
    Note that sym norm is used in the original GCN paper (kipf),
    while rw norm is used in GraphSAGE and some other variants.
    Here we don't perform sym norm since it doesn't seem to
    help with accuracy improvement.

    # Procedure:
    #       1. adj add self-connection --> adj'
    #       2. D' deg matrix from adj'
    #       3. norm by D^{-1} x adj'
    if sort_indices is True, we re-sort the indices of the returned adj
    Note that after 'dot' the indices of a node would be in descending order
    rather than ascending order
    """
    
    diag_shape = (adj.shape[0],adj.shape[1])
    D = adj.sum(1).flatten() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D,0),shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A

def add_self_loops(A, value=1.0):
    """Set the diagonal."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A

def totensor(features, labels):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.argmax(labels, -1))

    return features, labels  # , idx_train, idx_val, idx_test

def totensor_multi(features, labels):
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)

    return features, labels  # , idx_train, idx_val, idx_test


def preprocess_adj(adj):

    adj = adj + sp.eye(adj.shape[0])
    D1 = np.array(adj.sum(axis=1))**(-0.5)
    D2 = np.array(adj.sum(axis=0))**(-0.5)
    D1 = sp.diags(D1[:, 0], format='csr')
    D2 = sp.diags(D2[0, :], format='csr')

    A = adj.dot(D1)
    A = D2.dot(A)
    return A


def totensor(features, labels):
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.argmax(labels, -1))
    # A = sparse_mx_to_torch_sparse_tensor(adj)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    return features, labels  # , idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def to_numpy(x):
    """
    if isinstance(x, Variable):
        x = x.data
    """
    x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def calc_f1(y_true, y_pred,is_sigmoid=True):
    if not is_sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


"""
class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, input):
        value_dropped = F.dropout(input.storage.value(), self.p, self.training)
        return torch_sparse.SparseTensor(
                row=input.storage.row(), rowptr=input.storage.rowptr(), col=input.storage.col(),
                value=value_dropped, sparse_sizes=input.sparse_sizes(), is_sorted=True)
class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)
    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)
class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        if isinstance(input, torch_sparse.SparseTensor):
            res = input.matmul(self.weight)
            if self.bias:
                res += self.bias[None, :]
        else:
            if self.bias:
                res = torch.addmm(self.bias, input, self.weight)
            else:
                res = input.matmul(self.weight)
        return res
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)
def matrix_to_torch(X):
    if sp.issparse(X):
        return torch_sparse.SparseTensor.from_scipy(X)
    else:
        return torch.FloatTensor(X)
"""
