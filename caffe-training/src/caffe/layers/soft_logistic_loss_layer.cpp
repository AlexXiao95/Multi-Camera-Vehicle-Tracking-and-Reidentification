#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), bottom[0]->channels());
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
void SoftLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();//multidimensional label
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      float label = bottom_label[i * dim + j];
      Dtype prob = std::max(
        bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      loss -= label * log(prob);
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SoftLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* bottom_label = (*bottom)[1]->cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    int num = (*bottom)[0]->num();
    int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
    caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; j++) {
        float label = bottom_label[i * dim + j];
        Dtype prob = std::max(
          bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
        bottom_diff[i * dim + j] = label * scale / prob;
      }
    }
  }
}

INSTANTIATE_CLASS(SoftLogisticLossLayer);

}  // namespace caffe
