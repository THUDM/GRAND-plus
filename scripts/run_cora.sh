num_runs=$1
gpu=$2
prop_mat=$3
if [ $prop_mat = ppr ]
then
	#ppr
	python run_model.py --order 20 --prop_mode ppr --patience 200 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.5  --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.01 --input_droprate 0.5 --dataset cora --alpha 0.2 --warmup 1000 --rmax 1e-7 --seed2_runs $num_runs --cuda_device $gpu --clip-norm -1.0
elif [ $prop_mat = avg ]
then
	#avg
	python run_model.py --order 4 --prop_mode avg --patience 200 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.5  --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.01 --input_droprate 0.5 --dataset cora --warmup 1000 --rmax 1e-7 --seed2_runs $num_runs --cuda_device $gpu --clip-norm -1.0
elif [ $prop_mat = single ]
then
	#single
	python run_model.py --order 2  --prop_mode single --patience 200 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.5  --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.01 --input_droprate 0.5 --dataset cora --warmup 1000 --rmax 1e-7 --seed2_runs $num_runs --cuda_device $gpu --clip-norm -1.0
else
	echo Invalid propagation matrix: $prop_mat
fi
