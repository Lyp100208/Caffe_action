mkdir -p prob_seg
/usr/local/bin/mpirun -n 4 \
../../../tools/caffe_multigpu/build/tools/caffe_test_seg -model test_seg.prototxt -weights _iter_25000.caffemodel -folder prob_seg -output prob
