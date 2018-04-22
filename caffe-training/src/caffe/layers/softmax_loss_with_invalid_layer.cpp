#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {


template <typename Dtype>
void SoftmaxLossWithInvalidLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  this->softmax_layer_->Forward(this->softmax_bottom_vec_, &(this->softmax_top_vec_));
  const Dtype* prob_data = this->prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = this->prob_.num();
  int dim = this->prob_.count() / num;
  int spatial_dim = this->prob_.height() * this->prob_.width();
  Dtype loss = 0;
  int count = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      if (label[i * spatial_dim + j] >= 0.0) {
        loss -= log(std::max(prob_data[i * dim +
            static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
                           Dtype(FLT_MIN)));
        count++;
      }
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / count;
  if (top->size() == 2) {
    (*top)[1]->ShareData(this->prob_);
  }
}

template <typename Dtype>
void SoftmaxLossWithInvalidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = this->prob_.cpu_data();
    caffe_copy(this->prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = this->prob_.num();
    int dim = this->prob_.count() / num;
    int count = 0;
    int spatial_dim = this->prob_.height() * this->prob_.width();
    // LOG(INFO) << "the dim of p for softmax is:" << this->prob_.height() <<" "<<this->prob_.width() <<" "<<this->prob_.channels();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        if (label[i * spatial_dim + j] >= 0.0) {
          bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
              * spatial_dim + j] -= 1;
          ++count;
        } else {
          for (int c = 0; c < this->prob_.channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(this->prob_.count(), loss_weight / count, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLossWithInvalidLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLossWithInvalidLayer);


}  // namespace caffe
