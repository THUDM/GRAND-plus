python run_model.py --dataset aminer --use_bn --lr 0.01 --seed2_num 10 --seed1_num 10 --rmax 1e-5 --conf 0.1  --nlayers 1 --warmup 100 --prop_mode ppr --stop_mode acc --order 6 --pred_prop 6 --loss kl --weight_decay 1e-2 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 100 --batch_size 20 --alpha 0.1 --patience 10 --cuda_device 6
python run_model.py --dataset aminer --use_bn --lr 0.01 --seed2_num 10 --seed1_num 10 --rmax 1e-5 --conf 0.1  --nlayers 1 --warmup 100 --prop_mode avg --stop_mode acc --order 4 --pred_prop 4 --loss kl --weight_decay 1e-2 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 100 --batch_size 20 --alpha 0.1 --patience 10 --cuda_device 6
python run_model.py --dataset aminer --use_bn --lr 0.01 --seed2_num 10 --seed1_num 10 --rmax 1e-5 --conf 0.1  --nlayers 1 --warmup 100 --prop_mode single --stop_mode acc --order 2 --pred_prop 2 --loss kl --weight_decay 1e-2 --input_droprate 0.0 --hidden_droprate 0.0 --top_k 64 --lam 1.5 --unlabel_num 10000 --unlabel_batch_size 100 --batch_size 20 --alpha 0.1 --patience 10 --cuda_device 6
