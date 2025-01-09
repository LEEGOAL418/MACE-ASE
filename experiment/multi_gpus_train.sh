 #!/bin/bash   
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    ../mace/cli/run_train.py \
    --distributed \
    --config "./configs/LixC12_large.yaml"

