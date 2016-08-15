mkdir 5view
CUDA_VISIBLE_DEVICES=0,1,2,3 /usr/local/bin/mpirun -n 4 \
/data1/wangjiang/tools/caffe_multigpu/build/tools/caffe_test_multiseg_multiview -model test.prototxt -weights _iter_25000.caffemodel -output prob -folder 5view
