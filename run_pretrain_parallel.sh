#!/bin/sh

export PYTHONUNBUFFERED=1
export MASTER_PORT=14572

sudo apt-get update
sudo apt-get install fping -y

while ! fping $(seq $NSML_WORLD_SIZE | awk '{ print "node" $1-1 }'); do sleep 1; done

torchrun \
    --rdzv_id="${NSML_PROJECT}_${NSML_RUN}" \
    --rdzv_backend="c10d" \
    --rdzv_endpoint="${NSML_HOST_RANK0}:${MASTER_PORT}" \
    --nnodes=${NSML_WORLD_SIZE} \
    --nproc_per_node=${NSML_GPU_COUNT} \
    /mnt/mlx-nfs/CLUEGPT/PQGPT/main.py \
      --total_steps $total_steps \
      --session_type $session_type \
      --log_freq $log_freq \
      --test_freq $test_freq \
      --save_freq $save_freq \
      --lr $lr \
      --weight_decay $weight_decay \
      --max_norm $max_norm \
      --micro_batch_size $micro_batch_size \
      --global_batch_size $global_batch_size \
      --deepspeed $deepspeed \
      --num_workers $num_workers
