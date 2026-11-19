#!/bin/bash
# Adaptive PROBE: ~100 steps to verify mechanics + objective improvement
# NOT the real run (that's 1000 steps, initiated separately)
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=runs/probe_new/pc_adaptive
TARGET_STEPS=100
LOG="$OUT/run_adaptive.log"
mkdir -p "$OUT"

while true; do
    CURRENT=$(( $(wc -l < "$OUT/target0/metrics.csv" 2>/dev/null || echo 1) - 1 ))
    echo "[$(date)] Steps completed so far: $CURRENT / $TARGET_STEPS" | tee -a "$LOG"

    if [ "$CURRENT" -ge "$TARGET_STEPS" ]; then
        echo "[$(date)] Adaptive probe COMPLETE ($CURRENT steps)" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] Starting/resuming adaptive probe..." | tee -a "$LOG"
    $PY generation_finetuning.py \
        --ckpt_load_path runs/probe_new/pa_warmup/target0/ckpt_warmup_end.pt \
        --targets 0 --seed 99 \
        --arm adaptive \
        --n_steps 100 --warmup_steps 0 --stage_length 100 \
        --adaptive_min_phase 15 --adaptive_forced_intro 30 \
        --out "$OUT" \
        --resume \
        --probe 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Adaptive probe finished successfully" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] Process exited with code $EXIT_CODE, restarting in 10s..." | tee -a "$LOG"
    sleep 10
done
