#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TransformAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();
  LabelTransformParameter trans_param = this->layer_param_.label_transform_param();
  CHECK(trans_param.has_label_transform_file()) << this->layer_param_.name() << " No label transform provided";
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
void TransformAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[0]->count() / bottom[0]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void TransformAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  for (int i = 0; i < num; ++i) {
    // Top-k accuracy
    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j], j));
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
    // check if true label is in top k predictions
    int label_in = static_cast<int>(bottom_label[i]);
    int label_out = new_labels_[label_in];
    for (int k = 0; k < top_k_; k++) {
      if (bottom_data_vector[k].second == label_out) {
        ++accuracy;
        break;
      }
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(TransformAccuracyLayer);

}  // namespace caffe
