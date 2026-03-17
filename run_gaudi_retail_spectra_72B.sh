#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  SPECTRA + Qwen3-8B — Retail FULL BENCHMARK                    ║
# ║  Multi-agent architecture with deterministic guards + auditor   ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Same SPECTRA architecture as airline; retail has 115 tasks (vs 50).
#
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=tau-spectra-32B-retail
#SBATCH --output=/scratch/%u/tau-bench/logs/%x-%j.out
#SBATCH --error=/scratch/%u/tau-bench/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

CTR=/usr/bin/apptainer
CONTAINER="/data/sse/gaudi/containers/vllm-gaudi.sif"

# ── Paths ────────────────────────────────────────────────────────────
SCRATCH_BASE="/scratch/$USER"
TAU_DIR="/scratch/$USER/tau-bench"
mkdir -p "$SCRATCH_BASE"/{logs,habana_logs,home,.cache/huggingface}

export HOME="$SCRATCH_BASE/home"
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$SCRATCH_BASE/.cache"
export HABANA_LOGS="$SCRATCH_BASE/habana_logs"

# ── Model configuration ─────────────────────────────────────────────
AGENT_MODEL="Qwen/Qwen3-8B"
USER_MODEL="qwen3-30b-a3b-instruct-2507"
TP_SIZE=2
# ─────────────────────────────────────────────────────────────────────

PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "Launching Gaudi vLLM server on 127.0.0.1:${PORT} ..."
echo "  Model:  $AGENT_MODEL"
echo "  TP:     $TP_SIZE GPUs"

$CTR exec --writable-tmpfs \
  --bind /scratch:/scratch \
  --bind /data:/data \
  --bind "$HABANA_LOGS":"$HABANA_LOGS" \
  --env HABANA_VISIBLE_DEVICES=0,1 \
  --env HABANA_LOGS="$HABANA_LOGS" \
  --env HF_HOME="$HF_HOME" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env HOME="$HOME" \
  --env VLLM_SKIP_WARMUP=true \
  "$CONTAINER" \
  bash -c "pip install --no-cache-dir 'transformers>=4.51.0' && \
    vllm serve $AGENT_MODEL \
      --device hpu \
      --dtype bfloat16 \
      --block-size 128 \
      --max-model-len 16384 \
      --tensor-parallel-size $TP_SIZE \
      --port $PORT" \
  > "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM readiness (8B load may take 2-5 min)..."
for i in {1..600}; do
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM exited early. Tail of vLLM log:"
    tail -n 80 "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" || true
    exit 1
  fi
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready after ~$((i * 2))s."
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

# Load API keys for user model (Voyager)
if [ -f "$TAU_DIR/.env" ]; then
  set -a; source "$TAU_DIR/.env"; set +a
else
  echo "ERROR: $TAU_DIR/.env not found. Create it with your API keys." >&2; exit 1
fi

cd "$TAU_DIR"

VENV_PY="$SCRATCH_BASE/tau-bench-venv/bin/python"
RESULTS_DIR="$TAU_DIR/results/spectra-32B-retail"
mkdir -p "$RESULTS_DIR" "$TAU_DIR/logs"

"$VENV_PY" -c "import litellm; print('litellm ok')"

echo "============================================"
echo "  TAU-BENCH FULL RUN CONFIG"
echo "============================================"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Node:          $(hostname)"
echo "  Domain:        retail"
echo "  Strategy:      spectra (SPECTRA multi-agent)"
echo "  Agent model:   $AGENT_MODEL (local Gaudi vLLM, TP=$TP_SIZE)"
echo "  User model:    $USER_MODEL (Voyager)"
echo "  Num trials:    5 (pass^1 .. pass^5)"
echo "  Concurrency:   4"
echo "  Temperature:   0.7"
echo "  Results dir:   $RESULTS_DIR"
echo "============================================"

"$VENV_PY" run.py \
  --agent-strategy spectra \
  --env retail \
  --model "$AGENT_MODEL" \
  --model-provider openai \
  --user-model "$USER_MODEL" \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 0.7 \
  --max-concurrency 4 \
  --num-trials 5 \
  --log-dir "$RESULTS_DIR"

echo "============================================"
echo "  JOB COMPLETE: $(date)"
echo "  Results saved to: $RESULTS_DIR"
echo "============================================"

kill "$VLLM_PID" || true
