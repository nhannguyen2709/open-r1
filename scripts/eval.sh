# Default values matching those in vllm_infer_v2.py
LLM_MODEL_PTH=${1:-""}  # No default, this is required
MAX_NUM_SEQS=${2:-32}
MAX_MODEL_LEN=${3:-12282}
QUANTIZATION=${4:-"compressed-tensors"}

# Check if model path is provided (required parameter)
if [ -z "$LLM_MODEL_PTH" ]; then
    echo "Error: LLM model path is required as the first argument"
    exit 1
fi

# Extract model name (last part of the path)
MODEL_NAME=$(basename "$LLM_MODEL_PTH")

# Create output directory if it doesn't exist
mkdir -p generation

VLLM_WORKER_MULTIPROC_METHOD=spawn python src/open_r1/start_reward.py \
    --model "Qwen/Qwen2.5-Math-PRM-7B" \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.96 \
    --disable-frontend-multiprocessing \
    --distributed-executor-backend mp \
    --disable-log-requests \
    --disable-log-stats \
    --disable-fastapi-docs \
    --trust-remote-code \
    --swap-space 0 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --enforce-eager > rm_stdout.log 2> rm_stderr.log &
    # --override-pooler-config '{"softmax": false}'
# Capture the PID of the reward model process
REWARD_MODEL_PID=$!
echo "Started reward model with PID: $REWARD_MODEL_PID"
# Give the reward model a moment to initialize
sleep 5

for file in reference.csv aime-2024.csv aime-2025.csv; do
    # Create output filename from model name and CSV filename
    filename=$(basename "$file" .csv)
    output_file="generation/${MODEL_NAME}_${filename}.csv"
    
    echo "Processing $file with output to $output_file"
    
    python src/open_r1/vllm_infer_v3.py \
        --llm_model_pth $LLM_MODEL_PTH \
        --max_num_seqs $MAX_NUM_SEQS \
        --max_model_len $MAX_MODEL_LEN \
        --csv_file $file \
        --output_file $output_file \
        --quantization $QUANTIZATION
done
