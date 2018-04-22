#!/usr/bin/env sh

#GOOGLE_LOG_DIR=models/googlenet \
./build/tools/caffe train \
    --solver=models/new_googlenet/solver.prototxt \
    --gpu=3
