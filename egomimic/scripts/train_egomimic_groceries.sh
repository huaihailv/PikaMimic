export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 ./egomimic/scripts/pl_train.py --config ./egomimic/configs/egomimic_groceries.json 

