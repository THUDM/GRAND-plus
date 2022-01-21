from train_grandpp import *
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="grandpp",
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
    parser.add_argument('--weight_decay', type=float, default=5e-4,
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
    parser.add_argument('--dataset', type=str, default='reddit', help='Data set')
    parser.add_argument('--cuda_device', type=int,
                        default=2, help='Cuda device')
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
                        default=10, help='unlabel batch size')
    parser.add_argument('--eval_batch', type=int,
                        default=10, help='evaluation batch num')
    parser.add_argument('--batch_size', type=int,
                        default=5, help='batch size')
    parser.add_argument("--clip-norm", type=float,
                        default=-1, help="clip norm")
    parser.add_argument('--conf', type=float, default=0.5, help='confidence')
    parser.add_argument('--pred_prop', type=int, default=20, help='prop num')
    parser.add_argument('--walk_num', type=int,
                        default=0, help='random walk num')
    parser.add_argument('--unlabel_num', type=int,
                        default=0, help='unlabeled node ratio')
    parser.add_argument('--seed1_num', type=int, default=1,
                        help='unlabeled node ratio')
    parser.add_argument('--seed2_num', type=int, default=1,
                        help='unlabeled node ratio')
    parser.add_argument('--walk_mode', type=str,
                        default="online", help='online or index')
    parser.add_argument('--nlayers', type=int,
                        default=2, help='online or index')
    parser.add_argument('--stop_mode', type=str,
                        default='loss', help="acc, loss, both")
 
    args = parser.parse_args()
    print(args)
    time_total = []
    test_acc_total = []
    batch_time = []
    l1 = args.seed1_num  # 100
    l2 = args.seed2_num  # 20
    for s1 in range(l1):
        path = f"./{args.dataset}_output/"
        if not os.path.exists(path):
            print("{} don't exsits".format(path))
            os.mkdir(path)
            print("{} create success".format(path))
        else:
            print("{} exsits".format(path))
        args.seed1 = s1
        # time_tmp=[]
        # test_tmp=[]
        for s2 in range(l2):
            args.seed2 = s2
            t_total, test_acc, batch_time_av = main_grandpp(args)
            # time_tmp.append(t_total)
            # test_tmp.append(test_acc)
            time_total.append(t_total)
            batch_time.append(batch_time_av)
            test_acc_total.append(test_acc)
    print("time average", np.mean(time_total))
    print("test acc average", np.mean(test_acc_total))
    print("test acc std", np.std(test_acc_total))
    print("batch time average", np.mean(batch_time))
