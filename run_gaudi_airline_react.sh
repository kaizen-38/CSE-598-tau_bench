#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=tau-gaudi-airline-react
#SBATCH --output=/scratch/%u/tau-bench/logs/%x-%j.out
#SBATCH --error=/scratch/%u/tau-bench/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CTR=/usr/bin/apptainer
CONTAINER="/data/sse/gaudi/containers/vllm-gaudi.sif"

# Paths
SCRATCH_BASE="/scratch/$USER"
TAU_DIR="/scratch/$USER/tau-bench"
mkdir -p "$SCRATCH_BASE"/{logs,habana_logs,home,.cache/huggingface}

export HOME="$SCRATCH_BASE/home"
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$SCRATCH_BASE/.cache"

# Habana log path
export HABANA_LOGS="$SCRATCH_BASE/habana_logs"

# ── Model configuration ──────────────────────────────────────────────────
AGENT_MODEL="Qwen/Qwen3-32B"
USER_MODEL="qwen3-30b-a3b-instruct-2507"
# ─────────────────────────────────────────────────────────────────────────

PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "Launching Gaudi vLLM server on 127.0.0.1:${PORT} ..."
$CTR exec \
  --bind /scratch:/scratch \
  --bind /data:/data \
  --bind "$HABANA_LOGS":"$HABANA_LOGS" \
  --env HABANA_VISIBLE_DEVICES=0 \
  --env HABANA_LOGS="$HABANA_LOGS" \
  --env HF_HOME="$HF_HOME" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env HOME="$HOME" \
  --env VLLM_SKIP_WARMUP=true \
  "$CONTAINER" \
  vllm serve "$AGENT_MODEL" \
    --device hpu \
    --dtype bfloat16 \
    --block-size 128 \
    --max-model-len 16384 \
    --tensor-parallel-size 1 \
    --port "$PORT" \
  > "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM readiness..."
for i in {1..450}; do
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM exited early. Tail of vLLM log:"
    tail -n 80 "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" || true
    exit 1
  fi
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready."
    break
  fi
  sleep 2
done

if ! curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM never became ready. Tail of vLLM log:"
  tail -n 80 "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" || true
  exit 1
fi

export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
export OPENAI_API_KEY="EMPTY"

# Load API keys from .env (USER_MODEL_API_BASE, USER_MODEL_API_KEY, USER_MODEL_API_KEYS)
if [ -f "$TAU_DIR/.env" ]; then
  set -a; source "$TAU_DIR/.env"; set +a
else
  echo "ERROR: $TAU_DIR/.env not found. Create it with your API keys." >&2; exit 1
fi

cd "$TAU_DIR"

VENV_PY="$SCRATCH_BASE/tau-bench-venv/bin/python"
RESULTS_DIR="$TAU_DIR/results/gaudi-airline-react"
mkdir -p "$RESULTS_DIR" "$TAU_DIR/logs"

"$VENV_PY" -c "import litellm; print('litellm ok')"

echo "============================================"
echo "  TAU-BENCH RUN CONFIG"
echo "============================================"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Node:          $(hostname)"
echo "  Domain:        airline"
echo "  Strategy:      react"
echo "  Agent model:   $AGENT_MODEL (local Gaudi vLLM)"
echo "  User model:    $USER_MODEL (Voyager)"
echo "  Num trials:    5 (pass^1 .. pass^5)"
echo "  Concurrency:   6"
echo "  Temperature:   0.7"
echo "  Results dir:   $RESULTS_DIR"
echo "============================================"

"$VENV_PY" run.py \
  --agent-strategy react \
  --env airline \
  --model "$AGENT_MODEL" \
  --model-provider openai \
  --user-model "$USER_MODEL" \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 0.7 \
  --max-concurrency 6 \
  --num-trials 5 \
  --log-dir "$RESULTS_DIR"

echo "============================================"
echo "  JOB COMPLETE: $(date)"
echo "  Results saved to: $RESULTS_DIR"
echo "============================================"

kill "$VLLM_PID" || true
