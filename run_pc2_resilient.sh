#!/bin/bash
# Auto-resuming P-C.2 curriculum_pairs probe
# Restarts from checkpoint if the process dies for any reason
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=runs/probe_new/pc_curriculum_pairs
TARGET_STEPS=150
LOG="$OUT/run_pc2.log"

while true; do
    CURRENT=$(( $(wc -l < "$OUT/target0/metrics.csv" 2>/dev/null || echo 1) - 1 ))
    echo "[$(date)] Steps completed so far: $CURRENT / $TARGET_STEPS" | tee -a "$LOG"

    if [ "$CURRENT" -ge "$TARGET_STEPS" ]; then
        echo "[$(date)] P-C.2 COMPLETE ($CURRENT steps)" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] Starting/resuming P-C.2 curriculum_pairs..." | tee -a "$LOG"
    $PY generation_finetuning.py \
        --ckpt_load_path runs/probe_new/pa_warmup/target0/ckpt_warmup_end.pt \
        --targets 0 --seed 99 \
        --arm curriculum_pairs \
        --n_steps 150 --warmup_steps 0 --stage_length 150 \
        --out "$OUT" \
        --resume \
        --probe 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] P-C.2 finished successfully" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] Process exited with code $EXIT_CODE, restarting in 10s..." | tee -a "$LOG"
    sleep 10
done
