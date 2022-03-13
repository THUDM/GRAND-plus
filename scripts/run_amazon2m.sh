num_runs=$1
gpu=$2
prop_mat=$3
if [ $prop_mat = ppr ]
then
	#ppr
	python run_model.py --dataset Amazon2M --use_bn --node_norm --lr 1e-3 --seed2_runs 1 --seed1_runs $num_runs --hidden 1024 --rmax 1e-6 --nlayers 2 --warmup 500 --tem 0.1 --prop_mode ppr --stop_mode acc --order 6 --loss kl --weight_decay 1e-5 --clip-norm -1 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 0.8 --unlabel_num 10000 --unlabel_batch_size 200 --batch_size 50 --alpha 0.2 --patience 30 --cuda_device $gpu
elif [ $prop_mat = avg ]
then
	#avg
	python run_model.py --dataset Amazon2M --use_bn --node_norm --lr 1e-3 --seed2_runs 1 --seed1_runs $num_runs --hidden 1024 --rmax 1e-6 --nlayers 2 --warmup 500 --tem 0.1 --prop_mode avg --stop_mode acc --order 4 --loss kl --weight_decay 1e-5 --clip-norm -1 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 0.8 --unlabel_num 10000 --unlabel_batch_size 200 --batch_size 50 --patience 30 --cuda_device $gpu
elif [ $prop_mat = single ]
then
	#single
	python run_model.py --dataset Amazon2M --use_bn --node_norm --lr 1e-3 --seed2_runs 1 --seed1_runs $num_runs --hidden 1024 --rmax 1e-6 --nlayers 2 --warmup 500 --tem 0.1 --prop_mode single --stop_mode acc --order 2 --loss kl --weight_decay 1e-5 --clip-norm -1 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 32 --lam 0.8 --unlabel_num 10000 --unlabel_batch_size 200 --batch_size 50 --patience 30 --cuda_device $gpu
else
	echo Invalid propagation matrix: $prop_mat
fi
