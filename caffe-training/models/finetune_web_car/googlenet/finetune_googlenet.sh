MPI_PATH=/usr/bin/

#	--weights=googlenet_train_v4_iter_1000000.caffemodel \
# MPI implementation
GLOG_logtostderr=1 
$MPI_PATH/mpirun -np 2 ../../build/tools/caffe train \
	--solver=solver_googlenet.prototxt \
	--weights=googlenet_train_v4_iter_1000000.caffemodel \
2>&1 | tee googlenet_finetune_web_car.log 