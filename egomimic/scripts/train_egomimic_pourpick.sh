export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 ./egomimic/scripts/pl_train.py --config ./egomimic/configs/egomimic_pourpick.json 
        # --ckpt-path /home/lvhuaihai/EgoMimic/exp_egomimic_groceries_output_temporal_align/EgoMimic_test/human_groceries_4:1/models/last.ckpt

