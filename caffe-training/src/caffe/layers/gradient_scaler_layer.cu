#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();

    caffe_gpu_scale(count, Dtype(-coeff_), top_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS(GradientScalerLayer);

}  // namespace caffe
