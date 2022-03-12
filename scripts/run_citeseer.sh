num_runs=$1
gpu=$2
#ppr matrix
python run_model.py --stop_mode both --prop_mode ppr --order 10 --warmup 500 --tem 0.1 --lam 0.8 --top_k 32 --input_droprate 0.0 --hidden_droprate 0.0  --weight_decay 1e-3  --unlabel_batch_size 100 --batch_size 50 --patience 200 --hidden 256  --clip-norm -1 --sample 2 --alpha 0.4 --rmax 1e-7 --dataset citeseer --loss l2 --lr 0.001  --seed2_runs $num_runs --cuda_device $gpu
# avg matrix
python run_model.py --stop_mode both --prop_mode avg --order 2 --warmup 500 --tem 0.1 --lam 0.8 --top_k 32 --input_droprate 0.0 --hidden_droprate 0.0 --weight_decay 1e-3 --unlabel_batch_size 100 --batch_size 50 --patience 200 --hidden 256 --clip-norm -1  --sample 2 --rmax 1e-7 --dataset citeseer --loss l2 --lr 0.001 --seed2_runs $num_runs --cuda_device $gpu
# single matrix
python run_model.py --stop_mode both --prop_mode single --order 2 --warmup 500 --tem 0.1 --lam 0.8 --top_k 32 --input_droprate 0.0 --hidden_droprate 0.0 --weight_decay 1e-3 --unlabel_batch_size 100 --batch_size 50 --patience 200 --hidden 256 --clip-norm -1  --sample 2 --rmax 1e-7 --dataset citeseer --loss l2 --lr 0.001 --seed2_runs $num_runs --cuda_device $gpu
