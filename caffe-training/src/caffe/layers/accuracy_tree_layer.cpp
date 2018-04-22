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
void AccuracyTreeLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  TreeParameter tree_param = this->layer_param_.tree_param();
  AccuracyParameter accuracy_param = this->layer_param_.accuracy_param();
  top_k_ = accuracy_param.top_k();
  tree_depth_ = tree_param.tree_depth();
  depth_end_position_.push_back(0);

  CHECK(tree_param.label_transform_file_size() > 0) << this->layer_param_.name() << " No label transform file provided";
  // each node is a classifier
  num_nodes_ = tree_param.label_transform_file_size();
  int bottom_size = this->layer_param_.bottom_size();
  int top_size = this->layer_param_.top_size();
  // the last bottom is for the label
  CHECK_EQ(bottom_size, num_nodes_+1) << "labels should match with bottom blobs input";
  CHECK_EQ(top_size, tree_depth_) << "top num should match with tree depth";
  new_labels_.resize(num_nodes_);
  int to_read = 1;  // number to read in the current depth 
  int depth_level = 0;
  for (int t = 0; t < num_nodes_; ++t) {
    std::ifstream trans_file(tree_param.label_transform_file(t).c_str());
    CHECK(trans_file.is_open());
    int label_to;
    set<int> to;
    new_labels_[t].clear();
    while (trans_file >> label_to) {
      new_labels_[t].push_back(label_to);
      if (label_to >= 0) {
        to.insert(label_to);
      }
    }
    trans_file.close();
    num_classes_.push_back(to.size());
    
    LOG(INFO) << this->layer_param_.name() << ": Transform to "
        << to.size() << " unique labels";
    CHECK_EQ(to.size(), bottom[t]->count()/bottom[t]->num()) << this->layer_param_.name()
        << ": mismatched label sets and softmax dim";

    to_read -= 1;
    if (to_read == 0) {
      depth_end_position_.push_back(t+1);
      to_read = 0;
      depth_level ++;
      for (int tr = depth_end_position_[depth_level-1]; tr < depth_end_position_[depth_level]; ++tr)
        to_read += num_classes_[tr];
    }
  }
}

template <typename Dtype>
void AccuracyTreeLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // CHECK_EQ(bottom[0]->num(), bottom[1]->num())
  //    << "The data and label should have the same number.";
  CHECK_LE(top_k_, bottom[num_nodes_]->count() / bottom[num_nodes_]->num())
      << "top_k must be less than or equal to the number of classes.";
  CHECK_EQ(bottom[num_nodes_]->channels(), 1);
  CHECK_EQ(bottom[num_nodes_]->height(), 1);
  CHECK_EQ(bottom[num_nodes_]->width(), 1);

  for (int d = 0; d < tree_depth_; ++d) 
    (*top)[d]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void AccuracyTreeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  vector<Dtype> accuracy(tree_depth_, 0);
  const Dtype* bottom_label = bottom[num_nodes_]->cpu_data();
  int num = bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    int node_idx = 0; // classifier index
    int depth_level = 0; // depth level of the current classifier 

    // first get the accuracy of the top depth-1 layers
    while (depth_level < tree_depth_) {
      int raw_label = static_cast<int>(bottom_label[i]);
      int this_label = new_labels_[node_idx][raw_label]; 
      int dim = num_classes_[node_idx];
      const Dtype* bottom_data = bottom[node_idx]->cpu_data();

      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int j = 0; j < dim; ++j) {
        bottom_data_vector.push_back(
            std::make_pair(bottom_data[i * dim + j], j));
      }
      int this_top_k = 1;
      if (depth_level == tree_depth_ - 1)
         this_top_k = top_k_;
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + this_top_k,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      int k = 0;
      for ( k = 0; k < this_top_k; k++) {
        if (bottom_data_vector[k].second == this_label) {
          accuracy[depth_level] += 1;
          break;
        } 
      } 
      // if lower layers are wrong, break directly
      if (bottom_data_vector[k].second != this_label)
        break;

      // update the node_idx to the next level 
      int skip_nodes = 0;
      for (int node = depth_end_position_[depth_level]; node < node_idx; node++)
          skip_nodes += num_classes_[node]; 
      node_idx = depth_end_position_[depth_level+1] + skip_nodes + this_label;

      depth_level++;
    }
  }

  for (int d = 0; d < tree_depth_; ++d) 
    (*top)[d]->mutable_cpu_data()[0] = accuracy[d] / num;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyTreeLayer);

}  // namespace caffe
