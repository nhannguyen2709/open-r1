#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/open-r1/logs/%x-%j.out
#SBATCH --err=/fsx/open-r1/logs/%x-%j.err
#SBATCH --requeue

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.4

set -x -e

source ~/.bashrc
source openr1/bin/activate
echo Starting sglang server...

MODEL_ID=$1
PORT=$2
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
python3 -m sglang.launch_server --model-path $MODEL_ID --port=$PORT --skip-tokenizer-init --mem-fraction-static 0.6 --host=0.0.0.0 --dp-size=$NUM_GPUS