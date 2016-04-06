#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* const bottom_data,
    const int channels, const int height, const int width,
    const int kernel_size,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index of top blob
    const int c = index % channels;
    const int pn = index / channels;

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* bottom_slice =
        bottom_data + (kernel_size * pn * channels + c) * height * width;
    for (int n = kernel_size*pn; n < kernel_size*(pn+1); n++)
    {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          if (bottom_slice[h * width + w] > maxval) {
            maxidx = w + width * (h + height * (c + channels * n));
            maxval = bottom_slice[h * width + w];
          }
        }
      }
      bottom_slice += width * height * channels;
    }

    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads, const Dtype* const bottom_data,
    const int channels, const int height, const int width,
    const int kernel_size,
    Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

    //index of top blob
    const int c = index % channels;
    const int pn = index / channels;

    const Dtype* bottom_slice =
        bottom_data + (kernel_size * pn * channels + c) * height * width;
    Dtype cumsum = 0;
    for (int n = kernel_size*pn; n < kernel_size*(pn+1); n++)
    {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
           cumsum += 1 / (1 + exp(- bottom_slice[h * width + w]));
        }
      }
      bottom_slice += width * height * channels;
    }

    const Dtype thres = rand_idx[index] * cumsum;
    cumsum = 0;
	bottom_slice = bottom_data + (kernel_size * pn * channels + c) * height * width;    
    for (int n = kernel_size*pn; n < kernel_size*(pn+1); n++)
    {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          cumsum += 1 / (1 + exp(-bottom_slice[h * width + w]));
          if (cumsum >= thres) {
			rand_idx[index] = ((n * channels + c) * height + h) * width + w;
			top_data[index] = bottom_slice[h * width + w];
			return;
          }
        }
      }
      bottom_slice += width * height * channels;
    }
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads, const Dtype* const bottom_data,
    const int channels, const int height, const int width, 
    const int kernel_size,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index of top blob
    const int c = index % channels;
    const int pn = index / channels;

    const Dtype* bottom_slice =
        bottom_data + (kernel_size * pn * channels + c) * height * width;

    Dtype cumvalues = 0, cumsum = FLT_MIN;
    for (int n = kernel_size*pn; n < kernel_size*(pn+1); n++)
    {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
	        const float wei = 1 / (1 + exp(- bottom_slice[h * width + w]));
			cumsum += wei;
			cumvalues += bottom_slice[h * width + w] * wei;
        }
      }
      bottom_slice += width * height * channels;
    }
    top_data[index] = cumvalues / cumsum;
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* const bottom_data,
    const int channels, const int height, const int width, const int kernel_size,
    Dtype* const top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    //index of top blob
    const int c = index % channels;
    const int pn = index / channels;

    Dtype aveval = 0;
    const Dtype* bottom_slice =
        bottom_data + (kernel_size * pn * channels + c) * height * width;
    for (int n = kernel_size*pn; n < kernel_size*(pn+1); n++)
    {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          aveval += bottom_slice[h * width + w];
        }
      }
      bottom_slice += width * height * channels;
    }

    top_data[index] = aveval / (width * height * kernel_size);
  }
}

template <typename Dtype>
void PoolingNumLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_num_param().pool()) {
  case PoolingNumParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,
        channels_, height_, width_,
        kernel_size_,
        top_data, mask, top_mask);
    break;
  case PoolingNumParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, channels_, height_, width_,
        kernel_size_, rand_idx_.mutable_gpu_data(), top_data);
    } else {
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, channels_, height_, width_,
        kernel_size_, top_data);
    }
    break;
  case PoolingNumParameter_PoolMethod_AVE:
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,
        channels_, height_, width_,
        kernel_size_,
        top_data);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //top pixel index
    //const int c = index % channels;
    //const int pn = index / channels;

    int id_in_bottom;
    if (mask) {
      id_in_bottom = mask[index];
    } else {
      id_in_bottom = top_mask[index];
    }

    bottom_diff[id_in_bottom] = top_diff[index];
  }
}

template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads, const Dtype* const rand_idx,
    const Dtype* const top_diff, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //top pixel index
    //const int c = index % channels;
    //const int pn = index / channels;

    int id_in_bottom = rand_idx[index];
    bottom_diff[id_in_bottom] = top_diff[index];
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
        const int channels, const int height, const int width, const int kernel_size,
  Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    //top pixel index
    const int c = index % channels;
    const int pn = index / channels;

    Dtype* bottom_diff_slice =
        bottom_diff + (kernel_size * pn * channels + c) * height * width;
    const int pool_size = width * height * kernel_size;
    for (int n = kernel_size*pn; n < kernel_size*(pn+1); n++)
    {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
    bottom_diff_slice[h * width + w] = top_diff[index] / pool_size;
        }
      }
      bottom_diff_slice += width * height * channels;
    }
  }
}

template <typename Dtype>
void PoolingNumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();  
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const int count = top[0]->count();

  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_num_param().pool()) {
  case PoolingNumParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }

    // NOLINT_NEXT_LINE(whitespace/operators)    
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, bottom_diff);
    break;
  case PoolingNumParameter_PoolMethod_STOCHASTIC:
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, rand_idx_.gpu_data(), top_diff, bottom_diff);
    break;
  case PoolingNumParameter_PoolMethod_AVE:
     AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff,
        channels_, height_, width_, kernel_size_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingNumLayer);

}  // namespace caffe
