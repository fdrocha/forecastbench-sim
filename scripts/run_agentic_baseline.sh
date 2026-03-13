#!/usr/bin/env bash
# Agentic baseline experiment.
#
# Runs LLM agents (via pi) on CivBench forecasting surveys in an isolated
# sandbox. Each agent receives:
#   - AGENTS.md with domain knowledge + response format instructions
#   - Two survey files (seed0 = growth world, seed1 = crash world)
#   - Tools: read, bash, edit, write + web_search/fetch_content (via pi-web-access)
#
# The agent sees NO other context (no parent AGENTS.md, no skills/prompt-templates).
# Only the pi-web-access extension is loaded (for web search parity with human supers).
#
# Prerequisites:
#   - pi installed: npm install -g @mariozechner/pi-coding-agent
#   - pi-web-access installed: pi install npm:pi-web-access
#   - GCP credentials at ~/.config/gcloud/application_default_credentials.json
#   - civbench venv with google-cloud-secret-manager
#
# Usage:
#   cd /Users/elsehow/Projects/civbench
#   bash scripts/run_agentic_baseline.sh                          # all models, both surveys
#   bash scripts/run_agentic_baseline.sh claude-opus-4-6          # single model
#   bash scripts/run_agentic_baseline.sh claude-sonnet-4-5 seed0  # single model + seed

set -euo pipefail

CIVBENCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SANDBOX_DIR="/tmp/civbench-agentic-baseline"
RESULTS_DIR="$CIVBENCH_DIR/data/results/agentic_baseline"

MODELS=(
    openai/gpt-4.1
    anthropic/claude-sonnet-4-5
    openai/gpt-5
    google/gemini-3-pro-preview
    anthropic/claude-opus-4-6
)

SEEDS=(seed0 seed1 seed4 seed5 seed9 seed10 seed13 seed15 seed16 seed20)

# ── Parse args ─────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    MODELS=("$1")
fi
if [[ $# -ge 2 ]]; then
    SEEDS=("$2")
fi

# ── Load API keys from GCP Secret Manager ──────────────────────────────────
echo "Loading API keys from GCP..."
eval "$(cd "$CIVBENCH_DIR" && \
    GOOGLE_CLOUD_PROJECT=civbench \
    GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json" \
    uv run python -c "
from src.civrealm.evaluation.models import load_api_keys_from_gcp
import os
load_api_keys_from_gcp('civbench')
for k in ['ANTHROPIC_API_KEY','OPENAI_API_KEY','GEMINI_API_KEY']:
    v = os.environ.get(k,'')
    if not v and k=='GEMINI_API_KEY': v=os.environ.get('GOOGLE_API_KEY','')
    if v: print(f'export {k}={v}')
" 2>/dev/null)"

# ── Run each model ─────────────────────────────────────────────────────────
# Each run gets its own sandbox to prevent agents from modifying shared state
# (e.g., Opus overwrites input files with its responses).
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        model_short="${model##*/}"
        outfile="$RESULTS_DIR/response_${model_short}_${seed}.txt"
        run_sandbox="$SANDBOX_DIR/${model_short}_${seed}"
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "  Model: $model | Survey: $seed"
        echo "  Sandbox: $run_sandbox"
        echo "════════════════════════════════════════════════════════════"

        # Skip if response already exists (and has PERCENTILES)
        if [[ -f "$outfile" ]] && grep -q "<<<PERCENTILES>>>" "$outfile" 2>/dev/null; then
            echo "  → Already complete, skipping"
            continue
        fi
        # Also check sandbox file (Opus writes answers there)
        sandbox_file="$RESULTS_DIR/response_${model_short}_${seed}_sandbox.txt"
        if [[ -f "$sandbox_file" ]] && grep -q "<<<PERCENTILES>>>" "$sandbox_file" 2>/dev/null; then
            echo "  → Already complete (sandbox), skipping"
            continue
        fi

        rm -rf "$run_sandbox"
        mkdir -p "$run_sandbox"
        cp "$RESULTS_DIR/AGENTS.md" "$run_sandbox/"
        cp "$RESULTS_DIR/survey_${seed}.txt" "$run_sandbox/"

        cd "$run_sandbox"
        pi --model "$model" \
           --no-session \
           --no-skills \
           --no-prompt-templates \
           -p "Read survey_${seed}.txt and complete it per the instructions in AGENTS.md." \
           2>&1 | tee "$outfile"

        # Check if agent wrote answers to a sandbox file (Opus does this)
        for f in "$run_sandbox"/survey_*response*.txt "$run_sandbox"/survey_${seed}.txt; do
            if [[ -f "$f" ]] && grep -q "<<<PERCENTILES>>>" "$f" 2>/dev/null; then
                sandbox_dest="$RESULTS_DIR/response_${model_short}_${seed}_sandbox.txt"
                cp "$f" "$sandbox_dest"
                echo "  → Sandbox answers saved to $sandbox_dest"
                break
            fi
        done

        echo ""
        echo "  → Saved to $outfile"
    done
done

echo ""
echo "Done. Results in $RESULTS_DIR/response_*.txt"
