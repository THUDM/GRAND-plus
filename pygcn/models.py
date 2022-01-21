import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

import numpy as np
from pygcn.layers import GraphConvolution, MLPLayer

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):

        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc1(x, adj))
        # h1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = torch.log_softmax(x, dim=-1)
        return x
 
class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MLP, self).__init__()

        self.gc1 = MLPLayer(nfeat, nhid)
        self.gc2 = MLPLayer(nhid, nclass)
        #self.gc1 = torch.nn.Linear(nfeat, nhid)
        #self.gc2 = torch.nn.Linear(nhid, nclass)

        self.dropout = dropout
        # self.nonlinear = nn.SELU()

    def forward(self, x, adj):

        #x = F.dropout(x, self.dropout, training=self.training)
        #x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc1(x))
        # h1 = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        x = torch.log_softmax(x, dim=-1)
        return x
 