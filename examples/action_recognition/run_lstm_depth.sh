#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

#GLOG_logtostderr=1 \
#$TOOLS/caffe train -solver lstm_solver.prototxt -snapshot snapshots_lstm_32_iter_7000.solverstate \
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/openmpi-1.8.5/bin/mpirun -n 4 \
$TOOLS/caffe train -solver lstm_solver.prototxt -weights half_googlenet.caffemodel \
2>&1 | tee /data3/luozixin/aciton/log.txt
echo "Done."
