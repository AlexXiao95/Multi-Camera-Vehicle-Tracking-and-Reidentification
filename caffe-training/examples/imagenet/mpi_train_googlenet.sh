#!/usr/bin/env sh

#GOOGLE_LOG_DIR=models/googlenet/log \
mpirun -np 1 ./build/tools/caffe train \
    --solver=examples/imagenet/googlenet_solver.prototxt 2>&1 |tee /media/DataDisk/sqiu/models/googlenet/log/log.txt
#    --snapshot=models/googlenet/googlenet_train_iter_210000.solverstate
#    --gpu=3
