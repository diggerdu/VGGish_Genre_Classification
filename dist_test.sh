export WORLD_SIZE=$1
export TESTING_FLAG=1
python -m torch.distributed.launch --nproc_per_node=$1 main_test.py $2 $3
