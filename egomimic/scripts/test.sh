export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 ./egomimic/scripts/pl_train.py --config ./egomimic/configs/pikamimic_stackbasket.json 
