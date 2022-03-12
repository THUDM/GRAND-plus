from __future__ import division
from __future__ import print_function
import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
from precompute import propagation
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter
from utils.data_loader import load_data, accuracy


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, use_bn, input_dropout, hidden_dropout, node_norm):
        super().__init__()
        if nlayers == 1:
            self.embeds = torch.nn.Embedding(num_features, num_classes)
            self.fcs = nn.ModuleList([])
            self.bns = nn.ModuleList([])
        else:
            fcs = []
            bns = []
            self.embeds = torch.nn.Embedding(num_features, hidden_size)
            for i in range(nlayers - 2):
                fcs.append(nn.Linear(hidden_size, hidden_size, bias=True))
                bns.append(nn.BatchNorm1d(hidden_size))
            bns.append(nn.BatchNorm1d(hidden_size))
            fcs.append(nn.Linear(hidden_size, num_classes, bias=True))
            self.fcs = nn.ModuleList(fcs).cuda()
            self.bns = nn.ModuleList(bns).cuda()
        self.input_droprate = input_dropout
        self.hidden_droprate = hidden_dropout
        self.use_bn = use_bn
        self.node_norm = node_norm
        self.reset_param()

    def reset_param(self):
        for lin in self.fcs:
            lin.reset_parameters()

    def normalize(self, embedding):
        return embedding / (1e-12 + torch.norm(embedding, p=2, dim=-1, keepdim=True))

    def emb(self, attr_idx, node_idx, attr_data):
        feat_embeds = self.embeds(attr_idx).cuda()
        feat_embeds = F.dropout(feat_embeds, self.input_droprate, training=self.training)
        dim_size = node_idx[-1] + 1
        node_embeds = scatter(feat_embeds * attr_data[:, None], node_idx[:,None], dim=0, dim_size=dim_size, reduce='sum')
        node_s_sum = scatter(attr_data[:, None], node_idx[:, None], dim=0, dim_size=dim_size, reduce='sum')
        embs = node_embeds / (node_s_sum + 1e-10)
        return embs
        
    def forward(self, X):
        embs =  X
        for fc, bn in zip(self.fcs, self.bns):
            embs = F.relu(embs)
            if self.node_norm:
                embs = self.normalize(embs)
            if self.use_bn:
                embs = bn(embs)
            embs = F.dropout(embs, self.hidden_droprate, training=self.training)
            embs = fc(embs)
        return embs


class Grand_Plus(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, nlayers, use_bn, input_dropout, hidden_dropout, dropnode_rate=0.5, node_norm=False):
        super().__init__()
        self.mlp = MLP(num_features, num_classes,
                       hidden_size, nlayers, use_bn, input_dropout, hidden_dropout, node_norm = node_norm)
        self.dropnode_rate = dropnode_rate
    def forward(self, X):
        logits = self.mlp(X)
        return logits

    def random_prop(self, feats, mat_scores, mat_idx, dropnode_rate):
        mat_scores = F.dropout(mat_scores, p=dropnode_rate, training=self.training)
        propagated_logits = scatter(feats * mat_scores[:, None], mat_idx[:, None],
                                    dim=0, dim_size=mat_idx[-1] + 1, reduce='sum')
        mat_sum_s = scatter(mat_scores[:,None], mat_idx[:,None],
                                    dim=0, dim_size=mat_idx[-1] + 1, reduce='sum')
        return propagated_logits / (mat_sum_s + 1e-12)

    def emb(self, attr_idx, node_idx, attr_data, cuda= True):
        feat_embeds = self.mlp.emb(attr_idx, node_idx, attr_data)
        return feat_embeds

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


def consis_loss(args, logps, tem, conf):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)

    sharp_p = (torch.pow(avg_p, 1./tem) / torch.sum(torch.pow(avg_p, 1./tem), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        if args.loss == 'kl':
            loss += torch.mean((-sharp_p * torch.log(p)).sum(1)[avg_p.max(1)[0] > conf])
        elif args.loss == 'l2':
            loss += torch.mean((p-sharp_p).pow(2).sum(1)[avg_p.max(1)[0] > conf])
        else:
            raise ValueError(f"Unknown loss type: {args.loss}")
    loss = loss/len(ps)
    return loss


def valid(args, model, topk_adj, features, idx_val, labels, batch_size=100):
    model.eval()
    outputs = []
    for idx in iterate_minibatches_listinputs(idx_val, batch_size):
        val_topk_adj = topk_adj[idx]
        source_idx, neighbor_idx = val_topk_adj.nonzero()
        mat_scores = val_topk_adj.data
        val_feat = features[neighbor_idx]
        mat_scores = torch.tensor(mat_scores, dtype=torch.float32)
        source_idx = torch.tensor(source_idx, dtype=torch.long)
        y_val = labels[idx]
        node_idx, attr_idx = val_feat.nonzero()
        attr_data = val_feat.data
        attr_data = torch.tensor(attr_data, dtype=torch.float32)
        node_idx = torch.tensor(node_idx, dtype=torch.long)
        attr_idx = torch.tensor(attr_idx, dtype=torch.long)

        if args.cuda:
            mat_scores = mat_scores.cuda()
            source_idx = source_idx.cuda()
            node_idx = node_idx.cuda()
            attr_data = attr_data.cuda()

        with torch.no_grad():
            batch_emb = model.emb(attr_idx, node_idx, attr_data)
            batch_feat_aug = model.random_prop(batch_emb, mat_scores, source_idx, args.dropnode_rate)#.detach()
            output = model(batch_feat_aug)
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
            batch_attr = torch.FloatTensor(attr_mat[i:i + batch_size]).cuda()
            logits.append(model(batch_attr).to('cpu').numpy())
    logits = np.row_stack(logits)
    return logits


def predict(args, adj, features_np, model, idx_test, labels_org, mode='ppr', batch_size_logits=10000):
    model.eval()
    nprop = args.order
    embs, feats = [], []
    nnodes = features_np.shape[0]
    with torch.set_grad_enabled(False):
        for i in range(0, nnodes, batch_size_logits):
            batch_feat = features_np[i: i + batch_size_logits]
            node_idx, attr_idx = batch_feat.nonzero()
            attr_data = batch_feat.data
            attr_data = torch.tensor(attr_data, dtype=torch.float32).cuda()
            node_idx = torch.tensor(node_idx, dtype=torch.long).cuda()
            attr_idx = torch.tensor(attr_idx, dtype=torch.long)
            batch_embs = model.emb(attr_idx, node_idx, attr_data).to('cpu').numpy()
            embs.append(batch_embs)
    embs = np.row_stack(embs)
    
    if mode == 'ppr':
        embs = args.alpha * embs
        embs_prop = embs.copy()
        deg_row = adj.sum(1).A1
        deg_row_inv_alpha = np.asarray((1 - args.alpha) / np.maximum(deg_row, 1e-12))
        for _ in range(nprop):
            embs = np.multiply(deg_row_inv_alpha[:, None], (adj.dot(embs)))
            embs_prop += embs
        feats.append(embs_prop.copy())
    elif mode == 'avg':
        embs_prop = embs.copy()
        deg_row = adj.sum(1).A1
        deg_row_inv = 1 / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            embs =  np.multiply(deg_row_inv[:,None], (adj.dot(embs)))
            embs_prop += embs
        embs_prop = embs_prop/(nprop + 1)
        feats.append(embs_prop)
    elif mode == 'single':
        deg_row = adj.sum(1).A1
        deg_row_inv = 1 / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            embs = np.multiply(deg_row_inv[:,None], (adj.dot(embs)))
        feats.append(embs)
    else:
        raise ValueError(f"Unknown propagation mode: {mode}")
    for feat in feats:
        logits = get_local_logits(
            model.mlp, feat, batch_size_logits)    

        preds = logits.argmax(1)
        correct = np.equal(preds[idx_test], 
            labels_org.cpu().numpy()[idx_test]).astype(float)
        correct = correct.sum()
        acc_test = correct / len(idx_test)
        print(acc_test)
    return acc_test


def main_mag(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(args.cuda_device)
    device = args.cuda_device

    torch.manual_seed(args.seed2)
    if args.cuda:
        torch.cuda.manual_seed(args.seed2)
    np.random.seed(args.seed2)
    dataset = args.dataset

    adj, features, labels, idx_train, idx_val, idx_test, _ = load_data(dataset_str=dataset, split_seed=args.seed1)
    unlabel_num = args.unlabel_num
    time_s1 = time.time()
    adj = adj + sp.eye(features.shape[0])
    idx_sample = np.random.permutation(
        idx_test)[:unlabel_num]
    idx_unlabel = np.concatenate([idx_val, idx_sample]) 
    idx_train_unlabel = np.concatenate(
        [idx_train, idx_unlabel])
    
    indptr = np.array(adj.indptr, dtype=np.int32)
    indices = np.array(adj.indices, dtype=np.int32)
    graph = propagation.Graph(indptr, indices, args.seed2)
    row_idx = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.int32)
    col_idx = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.int32)
    mat_value = np.zeros((idx_train_unlabel.shape[0] * args.top_k), dtype=np.float64)
    if args.prop_mode == 'avg':
        coef = list(np.ones(args.order + 1, dtype=np.float64))
    elif args.prop_mode == 'ppr':
        coef = [args.alpha]
        for i in range(args.order):
            coef.append(coef[-1] * (1-args.alpha))
    elif args.prop_mode == 'single':
        coef = list(np.zeros(args.order + 1, dtype=np.float64))
        coef[-1] = 1.
    else:
        raise ValueError(f"Unknown propagation mode: {args.prop_mode}")
     
    print(f"propagation matrix: {args.prop_mode}")
    coef = np.asarray(coef) / np.sum(coef)
    graph.gfpush_omp(idx_train_unlabel, row_idx, col_idx, mat_value, coef, args.rmax, args.top_k)
    #print(row_idx.astype(np.int32).max(), col_idx.astype(np.int32).max(), features.shape[0])
    topk_adj = sp.coo_matrix((mat_value, (row_idx, col_idx)), (
        features.shape[0], features.shape[0]))
    topk_adj = topk_adj.tocsr()
    time_preprocessing = time.time() - time_s1
    print(f"preprocessing done, time: {time_preprocessing}")

    features_np = features
    n_class = labels.shape[1]
    labels = torch.LongTensor(np.argmax(labels, -1))
    
    model = Grand_Plus(num_features=features.shape[1],
                                num_classes=labels.max().item() + 1,
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
        labels = labels.cuda()

    t_begin = time.time()
    loss_values = []
    acc_values = []
    batch_time = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0
    loss_mn = np.inf
    acc_mx = 0.0
    best_epoch = 0
    num_batch = 0
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
            mat_scores = batch_topk_adj.data
            batch_feat = features[neighbor_idx]#.to(device)
            mat_scores = torch.tensor(mat_scores, dtype=torch.float32).to(device)
            source_idx = torch.tensor(source_idx, dtype=torch.long).to(device)
            y_train_batch = labels[train_index]
            node_idx, attr_idx = batch_feat.nonzero()
            attr_data = batch_feat.data
            attr_data = torch.tensor(attr_data, dtype=torch.float32).to(device)
            node_idx = torch.tensor(node_idx, dtype=torch.long).to(device)
            attr_idx = torch.tensor(attr_idx, dtype=torch.long)

            output_list = []
            K = args.sample
            loss_train = 0.
            for i in range(K):
                batch_emb = model.emb(attr_idx, node_idx, attr_data)
                batch_feat_aug = model.random_prop(batch_emb, mat_scores, source_idx, args.dropnode_rate)#.detach()
                output_aug = model(batch_feat_aug)
                output_aug = torch.log_softmax(output_aug, dim=-1)
                output_list.append(output_aug[len(train_index):])
                loss_train += F.nll_loss(output_aug[:len(train_index)], y_train_batch)

            loss_train = loss_train/K
            args.conf = 2./n_class
            loss_train += min(1.0 , float(num_batch)/args.warmup) * args.lam * consis_loss(args, output_list, args.tem, args.conf)

            acc_train = accuracy(output_aug[:len(train_index)], y_train_batch)
            loss_train.backward()
            grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)
            optimizer.step()
            batch_time.append(time.time() - batch_t_s)
            if num_batch % args.eval_batch == 0:
                loss_val, acc_val = valid(
                    args, model, topk_adj, features, idx_val, labels)

                loss_values.append(loss_val)
                acc_values.append(acc_val)
                if args.visible:
                    print(
                        f'epoch {epoch}, batch {num_batch}, validation loss {loss_val}, validation acc {acc_val}')
                 
                if acc_values[-1] >= acc_mx:
                    if args.stop_mode == 'acc' or (args.stop_mode == 'both' and loss_values[-1]<= loss_mn):
                        loss_mn = loss_values[-1]
                        acc_mx = acc_values[-1]
                        best_epoch = epoch
                        best_batch = num_batch
                        torch.save(model.state_dict(),
                                   f"{args.model}_{dataset}.pkl")
                        bad_counter = 0
                else:
                    bad_counter += 1
                if bad_counter >= args.patience:
                    if args.visible:
                        print(
                            f'Early stop! Min loss: {loss_mn}, Max accuracy: {acc_mx}, num batch: {num_batch} num epoch: {epoch}')
                    break
                    
            num_batch += 1
        if bad_counter >= args.patience:
            break
    if args.visible:
        print(
            f'Optimization Finished! Min loss: {loss_mn}, Max accuracy: {acc_mx}, num batch: {num_batch} num epoch: {epoch}')
    if args.visible:
        print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(f"{args.model}_{dataset}.pkl"))
    test_acc = predict(args, adj, features_np, model, idx_test, labels, mode = args.prop_mode)
    t_total = time.time() - time_s1
     
    print("Total time elapsed: {:.4f}s".format(t_total))
    print(f"Test Accuracy {test_acc}")

    return t_total, test_acc, np.mean(batch_time), num_batch


