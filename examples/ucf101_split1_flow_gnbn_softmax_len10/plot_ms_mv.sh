#./plot_response_ms_mv vidlist classlist root_img root_flow root_score root_vis num_seg num_views new_length seq_length stride_seq do_rolling
../../../tools/caffe_multigpu/build/tools/plot_response_ms_mv \
../../../datasets/ucf101_list/test_list_split1.txt \
../../../datasets/ucf101_list/ClassID_ucf101.txt \
../../../datasets/ucf101_img \
../../../data/ucf101_flow_tvl1_340_256 \
./prob_ms_mv \
./vis_prob \
25 10 10 1 1 0
