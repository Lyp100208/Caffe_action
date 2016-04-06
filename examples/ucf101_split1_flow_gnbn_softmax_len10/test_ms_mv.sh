mkdir -p prob_ms_mv
/usr/local/bin/mpirun -n 1 \
../../../tools/caffe_multigpu/build/tools/caffe_test_ms_mv -model test_ms_mv.prototxt -weights _iter_25000.caffemodel -folder prob_ms_mv -output prob -gpu -2
