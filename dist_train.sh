export WORLD_SIZE=$1
python -m torch.distributed.launch --nproc_per_node=$1 main.py $2
