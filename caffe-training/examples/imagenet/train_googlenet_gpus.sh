#!/usr/bin/env sh

GLOG_alsologtostderr=1 \
GOOGLE_LOG_DIR=models/googlenet \
./build/examples/parallel/gpus.bin \
    models/googlenet/solver.prototxt 1:2
