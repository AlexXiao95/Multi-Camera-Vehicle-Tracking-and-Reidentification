#!/usr/bin/env sh

#GOOGLE_LOG_DIR=models/googlenet \
./build/tools/caffe train \
    --solver=models/googlenet/solver.prototxt \
    --gpu=3
