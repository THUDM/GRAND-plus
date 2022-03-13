from model import main
from model_mag import main_mag
import os
import shutil
import argparse
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="grandpp",
                        help='model name')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training.')
    parser.add_argument('--cuda_device', type=int,
                        default=0, help='Cuda device')
    parser.add_argument('--seed1', type=int, default=42, help='split seed')
    parser.add_argument('--seed2', type=int, default=42, help='initialization seed')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--use_bn', action='store_true',
                        default=False, help='batch nmormalization') 
    parser.add_argument('--stop_mode', type=str,
        default='both', help="acc/both")
    parser.add_argument('--warmup', type=float,
                        default=1000, help='consistency loss warmup')
    parser.add_argument('--node_norm', action='store_true',
                        default=False, help='embedding L2 normalization')
    parser.add_argument("--clip-norm", type=float,
                        default=-1, help="clip norm")
    parser.add_argument('--eval_batch', type=int,
                        default=10, help='evaluation batch num')
    parser.add_argument('--batch_size', type=int,
                        default=50, help='batch size')
    parser.add_argument('--unlabel_batch_size', type=int,
                        default=100, help='unlabel batch size')
    parser.add_argument('--nlayers', type=int,
                        default=2, help='MLP layer num') 
    parser.add_argument('--hidden', type=int, default=64,
                        help='number of hidden units of MLP')
    parser.add_argument('--input_droprate', type=float, default=0.5,
                        help='dropout rate of the input layer (1 - keep probability)')
    parser.add_argument('--hidden_droprate', type=float, default=0.7,
                        help='dropout rate of the hidden layer (1 - keep probability)')
    parser.add_argument('--dropnode_rate', type=float, default=0.5,
                        help='dropnode rate (1 - keep probability)')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--sample', type=int, default=2,
                        help='augmentation times per batch')
    parser.add_argument('--tem', type=float, default=0.1,
                        help='sharpening temperature')
    parser.add_argument('--lam', type=float, default=1, help='Lamda')
    parser.add_argument('--alpha', type=float, default=0.2, help='ppr teleport')
    parser.add_argument('--top_k', type=int, default=32,
                        help='top neirghbor num')
    parser.add_argument('--rmax', type=float, default=1e-7,
                        help='GFPush threshold')
    parser.add_argument('--order', type=int, default=10, help='propagation step N')
    parser.add_argument('--unlabel_num', type=int,
            default=-1, help="unlabeled node num (|U'|) for consistency regularization")
    parser.add_argument('--prop_mode', type=str,
                        default="ppr", help='propagation matrix $\Pi$, ppr or avg or single')
    parser.add_argument('--loss', type=str,
                        default="l2", help="consistency loss function, l2 or kl")
    parser.add_argument('--seed1_runs', type=int, default=1,
                        help='data split runs')
    parser.add_argument('--seed2_runs', type=int, default=1,
                        help='model initialization runs')
    parser.add_argument('--visible', action='store_true',
                        default=False, help='batch nmormalization') 
     
    args = parser.parse_args()
    print(args)
    time_total = []
    test_acc_total = []
    batch_time = []
    batch_nums = []
    l1 = args.seed1_runs
    l2 = args.seed2_runs 
    for s1 in range(l1):
        args.seed1 = s1
        for s2 in range(l2):
            args.seed2 = s2
            if args.dataset == 'mag_scholar_c':
                t_total, test_acc, batch_time_av, batch_num = main_mag(args)
            else:
                t_total, test_acc, batch_time_av, batch_num = main(args)
            time_total.append(t_total)
            batch_time.append(batch_time_av)
            test_acc_total.append(test_acc)
            batch_nums.append(batch_num)
            print(f"split run: {s1}, initialization run: {s2}, avg acc: {np.mean(test_acc_total)}")
    print("time average", np.mean(time_total))
    print("test acc average", np.mean(test_acc_total))
    print("batch time average", np.mean(batch_time))
    print("batch num", np.mean(batch_nums))
