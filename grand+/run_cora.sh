python run_model.py --order 4 --pred_prop 4 --prop_mode avg --patience 100 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.5  --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.01 --input_droprate 0.5 --dataset cora --alpha 0.2 --warmup 1000 --eps 1e-7 --conf 0.3  --seed2_num 100 --cuda_device 1 --clip-norm -1.0
python run_model.py --order 10 --pred_prop 20 --prop_mode ppr --patience 100 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.5  --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.01 --input_droprate 0.5 --dataset cora --alpha 0.2 --warmup 1000 --eps 1e-7 --conf 0.3  --seed2_num 100 --cuda_device 1 --clip-norm -1.0
python run_model.py --order 2 --pred_prop 2 --prop_mode single --patience 100 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.5  --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.01 --input_droprate 0.5 --dataset cora --alpha 0.2 --warmup 1000 --eps 1e-7 --conf 0.3  --seed2_num 100 --cuda_device 1 --clip-norm -1.0
#python run_model.py --order 10 --pred_prop 20 --prop_mode ppr --patience 100 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.00 --batch_size 50 --unlabel_batch_size 100 --top_k 32 --stop_mode both --lr 0.02 --input_droprate 0.5 --dataset cora --alpha 0.2 --warmup 1000 --eps 1e-7 --conf 0.3  --seed2_num 1 --cuda_device 1 --clip-norm 1.0
#python run_model.py --order 2 --pred_prop 2 --prop_mode single --patience 100 --hidden_droprate 0.7 --hidden 64 --sample 2  --weight_decay 1e-3 --tem 0.1 --lam 1.00 --batch_size 50 --unlabel_batch_size 100  --top_k 32 --stop_mode both --lr 0.02 --input_droprate 0.5 --dataset cora --alpha 0.2 --warmup 1000 --eps 1e-7 --conf 0.3  --seed2_num 1 --cuda_device 1 --clip-norm 1.0
