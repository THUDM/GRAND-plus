from __future__ import division
from __future__ import print_function
import sys
sys.path.append("..")
import time
import argparse
import numpy as np
import scipy.sparse as sp
import propagation
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
from pygcn.utils import totensor
from collections import defaultdict
from torch.autograd import Variable
from torch_scatter import scatter
from torch.nn import init
from scipy.sparse import csr_matrix
from pygcn.utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor, preprocess_adj, normalize
from torch.nn.utils import weight_norm


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, use_bn, input_dropout, hidden_dropout, node_norm):
        super().__init__()
        if nlayers == 1:
            fcs = [nn.Linear(num_features, num_classes, bias=True)]
            bns = [nn.BatchNorm1d(num_features)]
        else:
            fcs = [nn.Linear(num_features, hidden_size, bias=True)]
            bns = [nn.BatchNorm1d(num_features)]

            for i in range(nlayers - 2):
                fcs.append(weight_norm(nn.Linear(hidden_size, hidden_size, bias=True)))
                bns.append(nn.BatchNorm1d(hidden_size))
            bns.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.Linear(hidden_size, num_classes, bias=True))
        # print(num_classes, "num_classes")
        
        self.fcs = nn.ModuleList(fcs)
        self.bns = nn.ModuleList(bns)
        self.input_droprate = input_dropout
        self.hidden_droprate = hidden_dropout
        self.use_bn = use_bn
        self.node_norm = node_norm
        # self.drop = Dropout(dropout)
        self.reset_param()

    def reset_param(self):
        for lin in self.fcs:
            lin.reset_parameters()

    def normalize(self, embedding):
        return embedding / (1e-12 + torch.norm(embedding, p=2, dim=-1, keepdim=True))

    def forward(self, X):
        if self.node_norm:
            X = self.normalize(X).detach()
        if self.use_bn:
            X = self.bns[0](X)
        embs = F.dropout(X, self.input_droprate, training=self.training)#.detach()
        embs = self.fcs[0](embs)
        
        for fc, bn in zip(self.fcs[1:], self.bns[1:]):
            embs = F.relu(embs)
            if self.node_norm:
                embs = self.normalize(embs)

            if self.use_bn:
                embs = bn(embs)
            embs = F.dropout(embs, self.hidden_droprate, training=self.training)

            embs = fc(embs)
            # print(embs.shape)
        # embs = self.normalize(embs)

        return embs


class Grand_MLP_Pre_Loss(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, use_bn, input_dropout, hidden_dropout, dropnode_rate=0.5, node_norm=False):
        super().__init__()
        self.mlp = MLP(num_features, num_classes,
                       hidden_size, nlayers, use_bn, input_dropout, hidden_dropout, node_norm = node_norm)
        self.dropnode_rate = dropnode_rate
    def forward(self, X):
        logits = self.mlp(X)
        #ppr_scores = F.dropout(ppr_scores, p = 0.5, training=self.training)
        # propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
        #                            dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        return logits

    def random_prop(self, feats, ppr_scores, ppr_idx, dropnode_rate):
        """
        if self.training:
            # noise = np.random.beta(a= a, b=b, size=ppr_scores.shape[0])
            ppr_scores = F.relu(torch.Tensor().to(ppr_scores.device)) * ppr_scores # / (a/(a+b))
        else:
            ppr_scores = ppr_scores
        """
        ppr_scores = F.dropout(ppr_scores, p=dropnode_rate, training=self.training)
        #feats = F.dropout(feats, p=0.5, training=self.training)
        # nnode = feats.shape[0]
        #print(feats.shape, ppr_scores.shape)
        propagated_logits = scatter(feats * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        ppr_sum_s = scatter(ppr_scores[:,None], ppr_idx[:,None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce='sum')
        return propagated_logits / (ppr_sum_s + 1e-12)


def iterate_minibatches_listinputs(index, batch_size, shuffle=False):
    numSamples = len(index)
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples, batch_size):
        if start_idx + batch_size > numSamples:
            end_idx = numSamples
        else:
            end_idx = start_idx + batch_size
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield index[excerpt]


def sample_unlabel(idx_unlabel, unlabel_batch_size, shuffle=False):
    unlabel_numSamples = idx_unlabel.shape[0]
    indices = np.arange(unlabel_numSamples)
    if shuffle:
        np.random.shuffle(indices)
    excerpt = indices[:unlabel_batch_size]
    return idx_unlabel[excerpt]


def clip_grad_norm(params, max_norm):
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None))


def consis_loss(args, logps, tem, lam, conf):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./tem) / torch.sum(torch.pow(avg_p, 1./tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        if args.loss == 'js':
            loss += torch.mean((-sharp_p * torch.log(p)).sum(1)[avg_p.max(1)[0] > conf])
        elif args.loss == 'l2':
            loss += torch.mean((p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
        else:
            raise ValueError(f"Unknown loss type: {args.loss}")
    loss = loss/len(ps)
    return loss


def valid(args, model, topk_adj, features, idx_val, labels, batch_size=10000):
    model.eval()
    outputs = []
    for idx in iterate_minibatches_listinputs(idx_val, batch_size):
        val_topk_adj = topk_adj[idx]
        source_idx, neighbor_idx = val_topk_adj.nonzero()
        ppr_scores = val_topk_adj.data
        val_feat = features[neighbor_idx]
        ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32)
        source_idx = torch.tensor(source_idx, dtype=torch.long)
        y_val = labels[idx]
        if args.cuda:
            val_feat = val_feat.cuda()
            ppr_scores = ppr_scores.cuda()
            source_idx = source_idx.cuda()
        with torch.no_grad():
            val_feat = model.random_prop(val_feat, ppr_scores, source_idx, args.dropnode_rate).detach()
            output = model(val_feat)
            # output = model.random_prop(output, ppr_scores, source_idx)
            output = torch.log_softmax(output, dim=-1)
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    loss_test = F.nll_loss(outputs, labels[idx_val])
    acc_test = accuracy(outputs, labels[idx_val])
    return loss_test.item(), acc_test.item()


def get_local_logits(model, attr_mat, batch_size=10000):
    device = next(model.parameters()).device
    nnodes = attr_mat.shape[0]
    logits = []
    with torch.set_grad_enabled(False):
        for i in range(0, nnodes, batch_size):
            batch_attr = torch.FloatTensor(attr_mat[i:i + batch_size]).to(device)
            #print(batch_attr.shape, i+batch_size)
            logits.append(model(batch_attr).to('cpu').numpy())
    logits = np.row_stack(logits)
    #print('logits', logits.shape)
    return logits


def predict(args, adj, features_np, model, idx_test, labels_org, mode='ppr', batch_size_logits=10000):
    model.eval()
    nprop = args.pred_prop

    #row, col = adj.nonzero()
    # print(adj)
    feats = []
    if mode == 'ppr':
        features_np = args.alpha * features_np
        features_np_prop = features_np.copy()
        # elif ppr_normalization == 'row':
        deg_row = adj.sum(1).A1
        deg_row_inv_alpha = np.asarray((1 - args.alpha) / np.maximum(deg_row, 1e-12))
        for _ in range(nprop):
            # print("adj", adj @ features_np)
            #features_np = (1 - args.alpha) * \
            #    np.multiply(deg_row_inv_alpha[:, None], (adj @ features_np))
            features_np = np.multiply(deg_row_inv_alpha[:, None], (adj.dot(features_np)))
            features_np_prop += features_np
        feats.append(features_np_prop.copy())
    elif mode == 'avg':
        features_np_prop = features_np.copy()
        deg_row = adj.sum(1).A1
        deg_row_inv = 1 / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            features_np =  np.multiply(deg_row_inv[:,None], (adj.dot(features_np)))
            features_np_prop += features_np
            # print(features_np_prop[0])
            #feats.append(features_np_prop.copy() / (_ + 2.))
        features_np_prop = features_np_prop/(nprop + 1)
        feats.append(features_np_prop)
    elif mode == 'single':
        #features_np_prop = features_np.copy()
        deg_row = adj.sum(1).A1
        deg_row_inv = 1 / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            features_np = np.multiply(deg_row_inv[:,None], (adj.dot(features_np)))
        features_np_prop = features_np
        feats = [features_np_prop]
    else:
        raise ValueError(f"Unknown propagation mode: {mode}")
    for feat in feats:
        # print(feat[0])
        logits = get_local_logits(
            model.mlp, feat, batch_size_logits)    

        preds = logits.argmax(1)
        # acc_test = accuracy(predictions[idx_test], labels[idx_test])
        # print(preds)
        correct = np.equal(preds[idx_test], 
            labels_org.cpu().numpy()[idx_test]).astype(float)
        correct = correct.sum()
        acc_test = correct / len(idx_test)
        print(acc_test)
    return acc_test


def main_grandpp(args):
    print(args.log)
    if args.log:
        logfile = open(f'{args.dataset}_output/{args.seed1}_{args.seed2}.txt', 'w')
    else:
        logfile = sys.stdout
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(args.cuda_device)
    device = args.cuda_device

    torch.manual_seed(args.seed2)
    if args.cuda:
        torch.cuda.manual_seed(args.seed2)
    np.random.seed(args.seed2)
    dataset = args.dataset
    
    adj, features, labels, idx_train, idx_val, idx_test, _ = load_data(
        dataset_str=dataset, split_seed=args.seed1)
    unlabel_num = args.unlabel_num
    time_s1 = time.time()
    adj = adj + sp.eye(features.shape[0])
    #features = adj.dot(features)/ (adj.sum(1) + 1e-10)
    idx_sample = np.random.permutation(
        idx_test)[:unlabel_num]
    idx_unlabel = np.concatenate([idx_val, idx_sample]) 
    idx_train_unlabel = np.concatenate(
        [idx_train, idx_unlabel])
    print(idx_train_unlabel.shape)
    indptr = np.array(adj.indptr, dtype=np.int32)
    indices = np.array(adj.indices, dtype=np.int32)
    graph = propagation.Graph(indptr, indices, args.seed2)
    row_idx = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.int32)
    col_idx = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.int32)
    ppr_value = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.float64)
    if args.prop_mode == 'avg':
        coef = list(np.ones(args.order + 1, dtype=np.float64))
    elif args.prop_mode == 'ppr':
        coef = [args.alpha]
        for i in range(args.order):
            coef.append(coef[-1] * (1-args.alpha))
    elif args.prop_mode == 'single':
        coef = list(np.zeros(args.order + 1, dtype=np.float64))
        coef[-1] = 1.0
    else:
        raise ValueError(f"Unknown propagation mode: {args.prop_mode}")
    coef = np.asarray(coef) / np.sum(coef)
    print('start rw')
    graph.forward_rw_omp(idx_train_unlabel, row_idx, col_idx, ppr_value, coef, args.eps, args.top_k)
    
    print(row_idx.astype(np.int32).max(), col_idx.astype(np.int32).max(), features.shape[0])
    topk_adj = sp.coo_matrix((ppr_value, (row_idx, col_idx)), (
        features.shape[0], features.shape[0]))
    topk_adj = topk_adj.tocsr()
    sum_s = np.asarray(topk_adj.sum(1)).squeeze()
    print('topk_adj: ', sum_s.shape, sum_s[idx_train_unlabel])
    time_preprocessing = time.time() - time_s1
    print(f"preprocessing done, time: {time_preprocessing}", file=logfile)
    features_np = features
    features, labels = totensor(features, labels)
    n_class = labels.max().item() + 1
    model = Grand_MLP_Pre_Loss(num_features=features.shape[1],
                                num_classes=n_class,
                                hidden_size=args.hidden,
                                nlayers=args.nlayers,
                                use_bn = args.use_bn,
                                input_dropout=args.input_droprate,
                                hidden_dropout=args.hidden_droprate, 
                                dropnode_rate=args.dropnode_rate,
                                node_norm = args.node_norm)
   
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    if args.cuda:
        model.cuda()
        labels = labels.cuda()

    t_begin = time.time()
    loss_values = []
    acc_values = []
    batch_time = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0
    loss_mn = np.inf
    acc_mx = 0.0
    best_epoch = 0
    num_batch = 0
    # print("prepare is over")
    for epoch in range(args.epochs):
        for train_index in iterate_minibatches_listinputs(idx_train, batch_size=args.batch_size, shuffle=True):
            batch_t_s = time.time()
            model.train()
            optimizer.zero_grad()
            unlabel_index_batch = sample_unlabel(
                    idx_sample, args.unlabel_batch_size, shuffle=True)
            batch_index = np.concatenate((train_index, unlabel_index_batch))
            batch_topk_adj = topk_adj[batch_index]

            source_idx, neighbor_idx = batch_topk_adj.nonzero()
            ppr_scores = batch_topk_adj.data
            #print('ppr_scores', ppr_scores.nonzero()[0].shape)
            # print('source idx shape ',source_idx.shape)
            # print('neighbor idx shape ', neighbor_idx.shape)
            # print('ppr score shape ', ppr_scores.shape)
            #print('neighbor idx shape ', neighbor_idx.shape, source_idx.shape)
            #print('ppr score shape ', ppr_scores.shape)
            batch_feat = features[neighbor_idx].to(device)
            ppr_scores = torch.tensor(ppr_scores, dtype=torch.float32).to(device)
            source_idx = torch.tensor(source_idx, dtype=torch.long).to(device)
            y_train_batch = labels[train_index]
            """
            if args.cuda:
                batch_feat = batch_feat.cuda()
                ppr_scores = ppr_scores.cuda()
                source_idx = source_idx.cuda()
            """
            output_list = []
            K = args.sample
            # print(output.shape, "output")
            loss_train = 0.
            for i in range(K):
                batch_feat_aug = model.random_prop(batch_feat, ppr_scores, source_idx, args.dropnode_rate).detach()
                output_aug = model(batch_feat_aug)
                output_aug = torch.log_softmax(output_aug, dim=-1)
                output_list.append(output_aug[len(train_index):])
                loss_train += F.nll_loss(output_aug[:len(train_index)], y_train_batch)
            #batch_feat_aug = model.random_prop(batch_feat, ppr_scores, source_idx, 0.0).detach()
            #output_aug = model(batch_feat_aug)
            #output_aug = torch.log_softmax(output_aug, dim=-1)        
            loss_train = loss_train/K
            """
            output_aug1 = torch.log_softmax(model.random_prop(
                output, ppr_scores, source_idx), dim=-1)
            output_aug2 = torch.log_softmax(model.random_prop(
                output, ppr_scores, source_idx), dim=-1)
            loss_train = 0.
            """
            #output = model(batch_feat, ppr_scores, source_idx)
            # print(output_aug1.shape)
            #loss_train += (F.nll_loss(output_aug1[:len(train_index)], y_train_batch) + F.nll_loss(
            #    output_aug1[:len(train_index)], y_train_batch))/2.
            args.conf = 2./n_class
            loss_train += min(args.lam, args.beta + ((args.lam - args.beta) * float(num_batch)/args.warmup)) * consis_loss(args, output_list, args.tem, args.lam,args.conf)

            acc_train = accuracy(output_aug[:len(train_index)], y_train_batch)
            loss_train.backward()
            grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)
            optimizer.step()
            batch_time.append(time.time() - batch_t_s)
            if num_batch % args.eval_batch == 0:
                loss_val, acc_val = valid(
                    args, model, topk_adj, features, idx_val, labels, args.batch_size)
                #loss_test, acc_test = valid(
                #    args, model, topk_adj, features, idx_test, labels)
                
                loss_values.append(loss_val)
                acc_values.append(acc_val)
                print(
                    f'epoch {epoch}, batch {num_batch}, validation loss {loss_val}, validation acc {acc_val}', file=logfile)
                
                #print(
                #    f'epoch {epoch}, batch {num_batch}, test loss {loss_test}, test acc {acc_test}', file=logfile)

                if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:
                    flag = False
                    if args.stop_mode == 'acc':
                        flag = acc_values[-1] >= acc_mx
                    elif args.stop_mode == 'loss':
                        flag = loss_values[-1] <= loss_mn
                    elif args.stop_mode == 'both':
                        flag = acc_values[-1] >= acc_mx and loss_values[-1] <= loss_mn
                    else:
                        flag = acc_values[-1] >= acc_mx or loss_values[-1] <= loss_mn

                    if flag:
                        loss_mn = loss_values[-1]
                        acc_mx = acc_values[-1]
                        best_epoch = epoch
                        best_batch = num_batch
                        torch.save(model.state_dict(),
                                   f"{args.model}_{dataset}.pkl")
                        # loss_mn = np.min((loss_values[-1], loss_mn))
                        # acc_mx = np.max((acc_values[-1], acc_mx))
                        bad_counter = 0
                else:
                    bad_counter += 1
                if bad_counter >= args.patience:
                    print(
                        f'Early stop! Min loss: {loss_mn}, Max accuracy: {acc_mx}, num batch: {num_batch} num epoch: {epoch}', file=logfile)
                    break

                    # return best_batch, best_epoch
            num_batch += 1
        if bad_counter >= args.patience:
            break

    print(
        f'Optimization Finished! Min loss: {loss_mn}, Max accuracy: {acc_mx}, num batch: {num_batch} num epoch: {epoch}', file=logfile)

   # Restore best model
    print('Loading {}th epoch'.format(best_epoch), file=logfile)
    model.load_state_dict(torch.load(f"{args.model}_{dataset}.pkl"))
    test_acc = predict(args, adj, features_np, model, idx_test, labels, mode = args.prop_mode)
    t_total = time.time() - time_s1
    print("Total time elapsed: {:.4f}s".format(t_total), file=logfile)
    print(f"Test Accuracy {test_acc}")
    return t_total, test_acc, np.mean(batch_time), num_batch


# Training settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="graphsage",
                        help='model name')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed1', type=int, default=42, help='split seed.')
    parser.add_argument('--seed2', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--input_droprate', type=float, default=0.5,
                        help='Dropout rate of the input layer (1 - keep probability).')
    parser.add_argument('--hidden_droprate', type=float, default=0.5,
                        help='Dropout rate of the hidden laayer (1 - keep probability).')
    parser.add_argument('--dropnode_rate', type=float, default=0.5,
                        help='Dropnode rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=20, help='Patience')
    parser.add_argument('--sample', type=int, default=2,
                        help='Sampling times of dropnode')
    parser.add_argument('--tem', type=float, default=0.1,
                        help='Sharpening temperature')
    parser.add_argument('--lam', type=float, default=1.0, help='Lamda')
    parser.add_argument('--dataset', type=str, default='cora', help='Data set')
    parser.add_argument('--cuda_device', type=int,
                        default=6, help='Cuda device')
    parser.add_argument('--alpha', type=float, default=0.1, help='Cuda device')
    parser.add_argument('--beta', type=float, default=0.0, help='Cuda device')
    parser.add_argument('--warmup', type=float,
                        default=100, help='Cuda device')
    parser.add_argument('--use_bn', type=bool,
                        default=False, help='Using Batch Normalization')
    parser.add_argument('--top_k', type=int, default=32,
                        help='top neirghbor num in ppr')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='ppr approximate')
    parser.add_argument('--unlabel_batch_size', type=int,
                        default=100, help='unlabel batch size')
    parser.add_argument('--eval_batch', type=int,
                        default=10, help='evaluation batch num')
    parser.add_argument('--batch_size', type=int,
                        default=10, help='batch size')
    parser.add_argument("--clip-norm", type=float,
                        default=1.0, help="clip norm")
    parser.add_argument('--conf', type=float, default=0.8, help='confidence')
    parser.add_argument('--pred_prop', type=int, default=5, help='prop num')
    parser.add_argument('--order', type=int, default=6, help='prop num')
    parser.add_argument('--unlabel_num', type=int,
                        default=0, help='unlabeled node ratio')
    parser.add_argument('--prop_mode', type=str,
                        default="ppr", help='ppr or avg')
    parser.add_argument('--nlayers', type=int,
                        default=2, help='layer num')
    parser.add_argument('--stop_mode', type=str,
                        default='loss', help="acc, loss, both")
    parser.add_argument('--log', type=bool,
                        default=True, help="acc, loss, both")
    parser.add_argument('--loss', type=str,
                        default="l2", help="l2, js")
    args = parser.parse_args()
    main_grandpp(args)
