#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/bvlc_reference_caffenet \
./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
    --gpu=2
