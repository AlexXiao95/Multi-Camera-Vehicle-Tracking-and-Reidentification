#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/googlenet \
./build/tools/caffe train \
    --solver=models/googlenet/solver.prototxt \
    --snapshot=models/googlenet/googlenet_train_iter_5000.solverstate \
    --gpu=3
