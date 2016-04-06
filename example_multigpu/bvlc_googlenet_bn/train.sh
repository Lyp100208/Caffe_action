#!/usr/bin/env sh

#GLOG_logtostderr=1 /usr/local/bin/mpirun -hostfile machine -n $1 ../../build/tools/caffe train --solver=solver.prototxt
GLOG_logtostderr=1 /usr/local/bin/mpirun -n $1 ../../build/tools/caffe train --solver=solver.prototxt
