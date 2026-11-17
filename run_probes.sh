#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

WARMUP_CKPT=runs/probe_new/pa_warmup/target0/ckpt_warmup_end.pt
if [ ! -f "$WARMUP_CKPT" ]; then
    echo "ERROR: warmup checkpoint not found at $WARMUP_CKPT"
    exit 1
fi

echo "=== Probe P-B: Mechanics test, compressed schedule ==="
# Start from warmup-end checkpoint, warmup_steps=0 (warmup already done)
for ARM in no_curriculum curriculum_weights curriculum_pairs adaptive; do
    echo ""
    echo "--- P-B: $ARM ---"
    $PY generation_finetuning.py \
        --ckpt_load_path "$WARMUP_CKPT" \
        --targets 0 --seed 99 \
        --arm $ARM \
        --n_steps 30 --warmup_steps 0 --stage_length 10 \
        --adaptive_min_phase 5 --adaptive_forced_intro 10 \
        --out "runs/probe_new/pb_mechanics/${ARM}" \
        --probe
    echo "P-B $ARM done"
done

echo ""
echo "--- P-B: Resume test (no_curriculum, kill after ~1 step then resume) ---"
$PY generation_finetuning.py \
    --ckpt_load_path "$WARMUP_CKPT" \
    --targets 0 --seed 99 \
    --arm no_curriculum \
    --n_steps 10 --warmup_steps 0 --stage_length 10 \
    --out "runs/probe_new/pb_mechanics/resume_test" \
    --probe &
PID=$!
sleep 150
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
echo "Killed, now resuming..."
$PY generation_finetuning.py \
    --ckpt_load_path "$WARMUP_CKPT" \
    --targets 0 --seed 99 \
    --arm no_curriculum \
    --n_steps 10 --warmup_steps 0 --stage_length 10 \
    --out "runs/probe_new/pb_mechanics/resume_test" \
    --resume \
    --probe
echo "P-B resume test done"

echo ""
echo "=== Probe P-C: Quality test, 150 steps from warmup checkpoint ==="
echo "--- P-C.1: no_curriculum ---"
$PY generation_finetuning.py \
    --ckpt_load_path "$WARMUP_CKPT" \
    --targets 0 --seed 99 \
    --arm no_curriculum \
    --n_steps 150 --warmup_steps 0 --stage_length 150 \
    --out runs/probe_new/pc_no_curriculum \
    --probe

echo ""
echo "--- P-C.2: curriculum_pairs ---"
$PY generation_finetuning.py \
    --ckpt_load_path "$WARMUP_CKPT" \
    --targets 0 --seed 99 \
    --arm curriculum_pairs \
    --n_steps 150 --warmup_steps 0 --stage_length 150 \
    --out runs/probe_new/pc_curriculum_pairs \
    --probe

echo ""
echo "=== All probes complete ==="
