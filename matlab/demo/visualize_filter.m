clear,clc
addpath('/data1/tools/caffe_12_12/matlab/');

% Set caffe mode
caffe.set_mode_cpu();

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
model_dir = '/data1/deep_action/models/ucf101_split1_flow_gnbn_bincls/';
net_model = [model_dir 'deploy.prototxt'];
net_weights = [model_dir '_iter_25000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
    error('weight file does not exist');
end
folder_dst = '/data1/deep_action/doc/firstfilter_ucf101_split1_flow_gnbn_bincls';
mkdir(folder_dst);

% Initialize a network
barwidth = 2;
net = caffe.Net(net_model, net_weights, phase);
W = net.layer_vec(1).params.get_data();
for n = 1:size(W, 4)
    W1 = W(:,:,:,n); %7*7*20
    img = zeros(7 * 2 + barwidth, 7 * 10 + barwidth * 9);
    
    cnt = 0;
    for x = 1:10
        for y = 1:2
            img((7+barwidth)*(y-1)+1 : (7+barwidth)*(y-1)+7, (7+barwidth)*(x-1)+1 : (7+barwidth)*(x-1)+7) = W1(:,:,cnt+1);
            cnt = cnt + 1;
        end
    end
    figure(1),clf
    imagesc(img);
    saveas(gcf, fullfile(folder_dst, sprintf('filter_%d.png', n)));
end