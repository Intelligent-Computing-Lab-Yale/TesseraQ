#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1

export HF_HUB_CACHE="/gpfs/gibbs/pi/panda/yl2447/llmc/cache"

llmc=/gpfs/gibbs/pi/panda/yl2447/llmc
export PYTHONPATH=$llmc:$PYTHONPATH

task_name=llama32_awq_tesseraq
#config=${llmc}/configs/quantization/llama3_2/awq_w2g128_L32_1b.yml
config=${llmc}/configs/quantization/llama3_2/tesseraq_w2g128_L32_1b.yml


nnodes=1
nproc_per_node=1


find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)


MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT

nohup \
torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${llmc}/llmc/__main__.py --config $config --task_id $task_id \
> ${task_name}.log 2>&1 &

sleep 2
ps aux | grep '__main__.py' | grep $task_id | awk '{print $2}' > ${task_name}.pid


