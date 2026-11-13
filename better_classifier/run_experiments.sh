#!/usr/bin/env bash
# Every run of the Task 4 comparison, in five parallel streams (the machine has 16 cores).
# One variant = baseline plus exactly one change; the important rows run with three seeds.
# Results land in ../_dev/runs/*.json; print the table afterwards with
#   uv run python train_leaves.py --summarise
#
# TORCH_HOME keeps the pretrained MobileNetV3 weights inside this project instead of the
# user profile. OMP_NUM_THREADS=4 because the streams share the machine. -X utf8 because
# python -I ignores PYTHONUTF8, and the ONNX export prints non-ASCII progress.
set -u
export PYTHONUTF8=1 TORCH_HOME=../_dev/torch_cache OMP_NUM_THREADS=4

run() { uv run python -I -X utf8 train_leaves.py --variant "$1" --seeds "$2"; }

run baseline 3 && run cosine 3 &           # the fixed baseline, then the schedule change
run augment 3  && run colour 3 &           # augmentation, then colour jitter on top of it
run frozen 3   && run combined 3 &         # only the last layer trained, then the combination
run full 3     && run augment_cosine 3 &   # the shipped stack, and the same minus colour jitter
run scratch 1  && run adamw 2 && run smooth 2 &
wait
echo "all runs finished"
