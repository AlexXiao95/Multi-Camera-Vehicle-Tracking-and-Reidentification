#include <cfloat>
#include <vector>
#include <fstream>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

template <typename Dtype>
void TransformSoftmaxWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  SoftmaxWithLossLayer<Dtype>::LayerSetUp(bottom, top);

  LabelTransformParameter trans_param = this->layer_param_.label_transform_param();
  CHECK(trans_param.has_label_transform_file()) << this->layer_param_.name() << " No label transform file provided";
  std::ifstream trans_file(trans_param.label_transform_file().c_str());
  CHECK(trans_file.is_open());
  int label_to;
  set<int> to;

  new_labels_.clear();
  while (trans_file >> label_to) {
      new_labels_.push_back(label_to);
      if (label_to >= 0) {
          to.insert(label_to);
      }
  }
  trans_file.close();

  LOG(INFO) << this->layer_param_.name() << ": Transform to "
          << to.size() << " unique labels";
  CHECK_EQ(to.size(), bottom[0]->count()/bottom[0]->num()) << this->layer_param_.name()
        << ": mismatched label sets and softmax dim";
}

template <typename Dtype>
void TransformSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  this->softmax_layer_->Forward(this->softmax_bottom_vec_, &(this->softmax_top_vec_));
  const Dtype* prob_data = this->prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = this->prob_.num();
  int dim = this->prob_.count() / num;
  int spatial_dim = this->prob_.height() * this->prob_.width();
  CHECK(this->prob_.height() == 1);
  CHECK(this->prob_.width() == 1);
  Dtype loss = 0;
  sample_cnt_ = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      int label_in = (int)(label[i * spatial_dim + j]);
      int label_out = new_labels_[label_in];
      sample_cnt_++;
      //std::cout << "(" << label_in << ", " << label_out << ") ";
      if (label_out >= 0) {
        loss -= log(std::max(prob_data[i * dim +
                              label_out * spatial_dim + j],
                             Dtype(FLT_MIN)));
      }
    }
  }
  //std::cout << std::endl;
  if (sample_cnt_ > 0)
    (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  else
    ; //(*top)[0]->mutable_cpu_data()[0] = 0;
  if (top->size() == 2) {
    (*top)[1]->ShareData(this->prob_);
  }
}

template <typename Dtype>
void TransformSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    int spatial_dim = this->prob_.height() * this->prob_.width();
    CHECK(this->prob_.height() == 1);
    CHECK(this->prob_.width() == 1);
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        int label_in = (int)(label[i * spatial_dim + j]);
        int label_out = new_labels_[label_in];
        if (label_out >= 0) {
          bottom_diff[i * dim + label_out
              * spatial_dim + j] -= 1;
        }
        else {
          // if we set the diff to zero, we assume the negatives have no gradient
          caffe_set(dim, Dtype(0), bottom_diff + i * dim);
          // otherwise, we should leave the bottom_diff unchanged
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(this->prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}
INSTANTIATE_CLASS(TransformSoftmaxWithLossLayer);

}
