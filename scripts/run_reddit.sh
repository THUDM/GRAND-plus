num_runs=$1
gpu=$2
prop_mat=$3
if [ $prop_mat = ppr ]
then
	# ppr
	python run_model.py --dataset reddit --use_bn --node_norm --lr 1e-4 --seed2_runs 1 --seed1_runs $num_runs --hidden 512 --rmax 1e-5 --nlayers 2 --warmup 500 --prop_mode ppr --stop_mode acc --order 6 --loss kl --weight_decay 0.0 --clip-norm 0.1 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 200 --batch_size 50 --alpha 0.05 --patience 20 --tem 0.1 --cuda_device $gpu
elif [ $prop_mat = avg ]
then
	# avg
	python run_model.py --dataset reddit --use_bn --node_norm --lr 1e-4 --seed2_runs 1 --seed1_runs $num_runs --hidden 512 --rmax 1e-5 --nlayers 2 --warmup 500 --prop_mode avg --stop_mode acc --order 6 --loss kl --weight_decay 0.0 --clip-norm 0.1 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 200 --batch_size 50 --patience 20 --tem 0.1 --cuda_device $gpu
elif [ $prop_mat = single ]
then
	#single
	python run_model.py --dataset reddit --use_bn --node_norm --lr 1e-4 --seed2_runs 1 --seed1_runs $num_runs --hidden 512 --rmax 1e-7 --nlayers 2 --warmup 500 --prop_mode single --stop_mode acc --order 2 --loss kl --weight_decay 0.0 --clip-norm 0.1 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 200 --batch_size 50 --patience 20  --tem 0.1 --cuda_device $gpu
else
	echo Invalid propagation matrix: $prop_mat
fi
