num_runs=$1
gpu=$2
prop_mat=$3
if [ $prop_mat = ppr ]
then
	#ppr
	python run_model.py --dataset aminer --use_bn --lr 0.01 --seed2_runs 1 --seed1_runs $num_runs --rmax 1e-5 --nlayers 1 --warmup 100 --prop_mode ppr --stop_mode acc --order 6  --loss kl --weight_decay 1e-2 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 100 --batch_size 20 --alpha 0.1 --patience 10 --cuda_device $gpu
elif [ $prop_mat = avg ]
then
	#avg
	python run_model.py --dataset aminer --use_bn --lr 0.01 --seed2_runs 1 --seed1_runs $num_runs --rmax 1e-5 --nlayers 1 --warmup 100 --prop_mode avg --stop_mode acc --order 4  --loss kl --weight_decay 1e-2 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 100 --batch_size 20 --patience 10 --cuda_device $gpu
elif [ $prop_mat = single ]
then
	#single
	python run_model.py --dataset aminer --use_bn --lr 0.01 --seed2_runs 1 --seed1_runs $num_runs --rmax 1e-5 --nlayers 1 --warmup 100 --prop_mode single --stop_mode acc --order 2  --loss kl --weight_decay 1e-2 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 100 --batch_size 20 --patience 10 --cuda_device $gpu
else
	echo Invalid propagation matrix: $prop_mat
fi
