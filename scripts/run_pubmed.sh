num_runs=$1
gpu=$2
prop_mat=$3
if [ $prop_mat = ppr ]
then
	#ppr
	python run_model.py --dataset pubmed --use_bn --node_norm --clip-norm 0.1 --stop_mode both --patience 100 --weight_decay 1e-2 --nlayers 1 --input_droprate 0.2 --hidden_droprate 0.2 --patience 50 --sample 2 --prop_mode ppr --alpha 0.5 --warmup 100 --top_k 16 --rmax 1e-5 --unlabel_batch_size 100 --batch_size 5 --order 6  --seed2_runs $num_runs --cuda_device $gpu
elif [ $prop_mat = avg ]
then
	#avg
	python run_model.py --dataset pubmed --use_bn --node_norm --clip-norm 0.1 --stop_mode both --patience 100 --weight_decay 1e-2 --nlayers 1 --input_droprate 0.2 --hidden_droprate 0.2 --patience 50 --sample 2 --prop_mode avg --warmup 1000 --top_k 16 --rmax 1e-5 --unlabel_batch_size 100 --batch_size 5 --order 4  --seed2_runs $num_runs --cuda_device $gpu
elif [ $prop_mat = single ]
then
	#single
	python run_model.py --dataset pubmed --use_bn --node_norm --clip-norm 0.1 --stop_mode both --patience 100 --weight_decay 1e-2 --nlayers 1 --input_droprate 0.2 --hidden_droprate 0.2 --patience 50 --sample 2 --prop_mode single --warmup 1000 --top_k 16 --rmax 1e-5 --unlabel_batch_size 100 --batch_size 5 --order 2  --seed2_runs $num_runs --cuda_device $gpu
else
	echo Invalid propagation matrix: $prop_mat
fi
