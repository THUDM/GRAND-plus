num_runs=$1
gpu=$2
prop_mat=$3
if [ $prop_mat = ppr ]
then
	#ppr
	python run_model.py --dataset mag_scholar_c --lr 0.01 --unlabel_num 10000 --stop_mode acc --weight_decay 0 --input_droprate 0.0 --hidden_droprate 0.2 --dropnode_rate 0.5 --patience 20 --sample 2 --alpha 0.2 --warmup 1000 --unlabel_batch_size 20 --eval_batch 10 --batch_size 20 --loss l2 --clip-norm -1 --order 10 --rmax 1e-5 --seed1_runs $num_runs --seed2_runs 1  --lam 1.0 --top_k 32 --prop_mode ppr --cuda_device $gpu
elif [ $prop_mat = avg ]
then
	#avg
	python run_model.py --dataset mag_scholar_c --lr 0.01 --unlabel_num 10000 --stop_mode acc --weight_decay 0 --input_droprate 0.0 --hidden_droprate 0.2 --dropnode_rate 0.5 --patience 20 --sample 2 --warmup 1000 --unlabel_batch_size 20 --eval_batch 10 --batch_size 20 --loss l2 --clip-norm -1 --order 10 --rmax 1e-5 --seed1_runs $num_runs --seed2_runs 1  --lam 1.0 --top_k 32 --prop_mode avg --cuda_device $gpu
elif [ $prop_mat = single ]
then
	#single
	python run_model.py --dataset mag_scholar_c --lr 0.01 --unlabel_num 10000 --stop_mode acc --weight_decay 0 --input_droprate 0.0 --hidden_droprate 0.2 --dropnode_rate 0.5 --patience 20 --sample 2 --warmup 1000 --unlabel_batch_size 20 --eval_batch 10 --batch_size 20 --loss l2 --clip-norm -1 --order 2 --rmax 1e-5 --seed1_runs $num_runs --seed2_runs 1  --lam 1.0 --top_k 32 --prop_mode single --cuda_device $gpu
else
	echo Invalid propagation matrix: $prop_mat
fi
