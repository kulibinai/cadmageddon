#!/bin/bash
# Reference: how the model was served for the paper's runs (vLLM, OpenAI API).
#
#     ./serving/serve_vllm.sh                       # 1 GPU, port 8000
#     TP=4 GPUS=0,1,2,3 ./serving/serve_vllm.sh     # the paper: 4 GPUs
#     GPUS=1 PORT=8001 ./serving/serve_vllm.sh      # a second server on GPU 1
#
# then point the agent at it:
#
#     python agent_run.py --mesh part.stl --out runs/ \
#         --base-url http://127.0.0.1:8000/v1 --model Qwen3.8-27B
#
# Everything is an environment variable; the defaults are the settings we used.
#   VLLM               vllm executable                        (vllm on PATH)
#   MODEL              Hugging Face model id                  (Qwen/Qwen3.8-27B-FP8)
#   SERVED_NAME        name the agent passes as --model       (Qwen3.8-27B)
#   PORT               HTTP port                              (8000)
#   TP                 tensor-parallel size                   (1; the paper used 4)
#   GPUS               CUDA_VISIBLE_DEVICES, TP devices       (0..TP-1)
#   MAX_MODEL_LEN      context length in tokens               (250000)
#   MAX_IMG_PER_PROMPT images allowed in one prompt           (12)
#   GPU_MEM_UTIL       --gpu-memory-utilization               (0.90)
#   SERVE_LOG          log file; keep it on LOCAL disk        (./vllm_serve.log)
#   REASONING_PARSER   --reasoning-parser, e.g. qwen3         (off)
#   SPECULATIVE_CONFIG --speculative-config, e.g.             (off)
#                      '{"method":"mtp","num_speculative_tokens":3}'
#   MAX_NUM_BATCHED_TOKENS --max-num-batched-tokens           (vLLM default)
#   HF_HOME            weight cache (standard HF variable)    (HF default)
#
# The FP8 weights are ~30 GB, so TP=1 on a single 48 GB+ GPU also works; more
# GPUs buy decode speed, not the ability to run.
#
# REASONING_PARSER changes what the agent sees. Without it the model's thinking
# comes back inline in `content`, which agent_run.py keeps in the history and
# compacts; with it the thinking moves to `reasoning_content`, which agent_run.py
# drops. MTP speculative decoding (SPECULATIVE_CONFIG) only changes speed; with
# it, raise MAX_NUM_BATCHED_TOKENS (e.g. 65536): it also sizes the image encoder
# cache, and the drafter can miss evicted image embeddings in a small one.
#
# Keep SERVE_LOG on local disk: a server writing its log to a network mount
# that hangs blocks on its own log pages and cannot be killed.
set -euo pipefail
VLLM=${VLLM:-vllm}
MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}
SERVED_NAME=${SERVED_NAME:-Qwen3.8-27B}
PORT=${PORT:-8000}
TP=${TP:-1}
GPUS=${GPUS:-$(seq -s, 0 $((TP - 1)))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-250000}
MAX_IMG_PER_PROMPT=${MAX_IMG_PER_PROMPT:-12}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
LOG=${SERVE_LOG:-$PWD/vllm_serve.log}
EXTRA=()
[ -n "${REASONING_PARSER:-}" ] && EXTRA+=(--reasoning-parser "$REASONING_PARSER")
[ -n "${SPECULATIVE_CONFIG:-}" ] && EXTRA+=(--speculative-config "$SPECULATIVE_CONFIG")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && EXTRA+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
export CUDA_VISIBLE_DEVICES=$GPUS
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}

if [ "$(tr ',' '\n' <<<"$GPUS" | grep -c .)" -ne "$TP" ]; then
  echo "serve_vllm.sh: GPUS=$GPUS does not name TP=$TP devices" >&2; exit 1
fi

echo "serving $MODEL as $SERVED_NAME on :$PORT, TP=$TP on GPUs $GPUS, log -> $LOG"
exec "$VLLM" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --limit-mm-per-prompt '{"image": '"$MAX_IMG_PER_PROMPT"'}' \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"} >>"$LOG" 2>&1
