/usr/local/bin/mpirun -n 4 \
/data1/wangjiang/tools/caffe_act/build/tools/caffe train -solver solver.prototxt -weights stage1_iter_10000_split1.caffemodel 2>&1 | tee log.txt
