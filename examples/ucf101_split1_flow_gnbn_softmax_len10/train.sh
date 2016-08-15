/usr/local/bin/mpirun -n 4 \
/data1/wangjiang/tools/caffe_act/build/tools/caffe train -solver solver.prototxt -weights /data1/wangjiang/deep_action/models/googlenet_bn/googlenet_bn_fold_20.caffemodel 2>&1 | tee log.txt
