#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/vgg \
./build/tools/caffe train \
    --solver=models/vgg/solver.prototxt \
    --gpu=1
