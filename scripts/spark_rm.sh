#!/bin/bash
# MODEL=<hf id> sbatch -N <nodes> -t <time> -J <name> scripts/spark_rm.sh ARM OUT [args...]
# One GB10 per node, torchrun data parallel; the repo is synced to /scratch/scholar/manors/nuance/code.
#SBATCH -p spark-batch -A scholar --gpus-per-node=1 --ntasks-per-node=1 --cpus-per-task=20
#SBATCH -o /scratch/scholar/manors/nuance/logs/%x-%j.log
cd /scratch/scholar/manors/nuance/code
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
M=models--${MODEL//\//--}
# the weights load far faster from the node's NVMe than over NFS: copy them to each node first
t=$(date +%s)
srun --ntasks-per-node=1 bash -c "mkdir -p /tmp/manors-hf/hub && rm -rf /tmp/manors-hf/hub/$M && cp -rL /scratch/scholar/manors/nuance/hf/hub/$M /tmp/manors-hf/hub/"
echo "staged $MODEL to /tmp on every node ($(( $(date +%s) - t )) s)"
# triton compiles into a per-node cache: 16 nodes writing one NFS cache race into stale file handles
export TRITON_CACHE_DIR=/tmp/manors-triton HF_HOME=/tmp/manors-hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
srun ../venv/bin/torchrun --nnodes=$SLURM_NNODES --nproc_per_node=1 --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29511 scripts/spark_rm.py "$@" --model $MODEL
