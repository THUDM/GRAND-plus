import math

import torch

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class MLPLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.bn = nn.BatchNorm1d(out_features, affine = False, eps=1e-6, momentum = 0.1)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # if out_features == 16:
        #     self.weight.requires_grad = False
            # self.weight0 = Parameter(torch.FloatTensor(out_features, out_features))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        #output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.bn = nn.BatchNorm1d(out_features, affine = False, eps=1e-6, momentum = 0.1)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # if out_features == 16:
        #     self.weight.requires_grad = False
            # self.weight0 = Parameter(torch.FloatTensor(out_features, out_features))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # print(adj, support)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class DGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, concat, bias=False):
        super(DGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a1 = Parameter(torch.FloatTensor(out_features, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)

        self.a2 = Parameter(torch.FloatTensor(out_features, 1))
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)

        self.concat = concat

        # self.bn = nn.BatchNorm1d(out_features, affine = False, eps=1e-6, momentum = 0.1)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features, 1))
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

        self.special_spmm = SpecialSpmm()
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, adj):
        support = torch.mm(input, self.W)

        n = input.size(0)
        l_D = F.sigmoid(self.leakyrelu(torch.mm(support, self.a1).squeeze()))
        r_D = F.sigmoid(self.leakyrelu(torch.mm(support, self.a2).squeeze()))

        ind_D = torch.LongTensor([range(n), range(n)])

        # L_D = torch.sparse.FloatTensor(ind_D, l_D, torch.Size([n, n]))
        # R_D = torch.sparse.FloatTensor(ind_D, r_D, torch.size([n, n]))
        output = self.special_spmm(ind_D, r_D, torch.Size([n, n]), support)
        output = torch.spmm(adj, output)
        output = torch.spmm(adj, output)
        output = torch.spmm(adj, output)
        output = self.special_spmm(ind_D, l_D, torch.Size([n, n]), output)


        sumnorm = self.special_spmm(ind_D, r_D, torch.Size([n, n]), torch.ones(size=(n,1)))
        sumnorm = torch.spmm(adj, sumnorm)
        sumnorm = self.special_spmm(ind_D, l_D, torch.Size([n, n]), sumnorm)

        output = output.div(sumnorm)

        if self.concat:
            # if this layer is not last layer,
            output = F.elu(output)
            # if this layer is last layer,


        if self.bias is not None:
            return output + self.bias.squeeze()
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
