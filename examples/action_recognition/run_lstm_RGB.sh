#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

#GLOG_logtostderr=1 \
#$TOOLS/caffe train -solver lstm_solver.prototxt -snapshot snapshots_rgb_iter_1000.solverstate \
CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 \
$TOOLS/caffe train -solver lstm_solver.prototxt -weights rgb.caffemodel \
2>&1 | tee log.txt
echo "Done."
