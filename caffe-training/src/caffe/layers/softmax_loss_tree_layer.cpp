#include <cfloat>
#include <vector>
#include <fstream>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

template <typename Dtype>
void SoftmaxWithLossTreeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // make a buffer
  vector<Dtype> lw_buffer;
  for (int t = 0; t < top->size(); ++t)
    lw_buffer.push_back(this->layer_param_.loss_weight(t));
  this->layer_param_.clear_loss_weight();
  //

  TreeParameter tree_param = this->layer_param_.tree_param();
  tree_depth_ = tree_param.tree_depth();
  depth_end_position_.push_back(0);

  CHECK(tree_param.label_transform_file_size() > 0) << this->layer_param_.name() << " No label transform file provided";
  // each node is a classifier
  num_nodes_ = tree_param.label_transform_file_size();
  CHECK(tree_param.node_weight_size() <= num_nodes_) << this->layer_param_.name() << " node_weight should not be bigger than node size";
  for (int t = 0; t < tree_param.node_weight_size(); ++t) {
    node_weight_.push_back(tree_param.node_weight(t));
  }
  while (node_weight_.size() < num_nodes_)
    node_weight_.push_back(Dtype(1.0));

  int bottom_size = this->layer_param_.bottom_size();
  int top_size = this->layer_param_.top_size();
  softmax_bottom_vec_.resize(num_nodes_);
  softmax_top_vec_.resize(num_nodes_);
  softmax_layer_.resize(num_nodes_);
  prob_.resize(num_nodes_);
  // TODO: add layer_weight here
  CHECK_EQ(bottom_size, num_nodes_+1) << "labels should match with bottom blobs input";
  CHECK_EQ(top_size, tree_depth_) << "top num should match with tree depth";
  new_labels_.resize(num_nodes_);
  node_loss_.resize(num_nodes_);
  int to_read = 1; // number to read in the current depth
  int depth_level = 0; // current depth level
  for (int t = 0; t < num_nodes_; ++t) {
    softmax_bottom_vec_[t].clear();
    softmax_top_vec_[t].clear();
    prob_[t].reset(new Blob<Dtype>());
    softmax_layer_[t].reset(new SoftmaxLayer<Dtype>(this->layer_param_));
    // read the label file
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
    // construct a tree
    to_read -= 1; 
    if (to_read == 0) {
      depth_end_position_.push_back(t+1);
      to_read = 0;
      depth_level ++;
      for (int tr = depth_end_position_[depth_level-1]; tr < depth_end_position_[depth_level]; ++tr)
        to_read += num_classes_[tr];
    }
    // setup softmax layer
    softmax_bottom_vec_[t].push_back(bottom[t]);
    softmax_top_vec_[t].push_back(prob_[t].get());
    softmax_layer_[t]->SetUp(softmax_bottom_vec_[t], &softmax_top_vec_[t]);
  }
  for (int t = 0; t < top->size(); ++t)
    this->layer_param_.add_loss_weight(lw_buffer[t]);
}

template <typename Dtype>
void SoftmaxWithLossTreeLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  for ( int t = 0; t < num_nodes_; ++t) {
    softmax_layer_[t]->Reshape(softmax_bottom_vec_[t], &softmax_top_vec_[t]);
    // Don't need this. Deprecated.
    //if (top->size() >= 2) {
      // softmax output
     // (*top)[1]->ReshapeLike(*bottom[0]);
   // }
  }
  for (int d = 0; d < tree_depth_; ++d) {
    (*top)[d]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void SoftmaxWithLossTreeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  for ( int t = 0; t < num_nodes_; ++t) {
    softmax_layer_[t]->Forward(softmax_bottom_vec_[t], &(softmax_top_vec_[t]));
    const Dtype* prob_data = prob_[t]->cpu_data();
    const Dtype* label = bottom[num_nodes_]->cpu_data();
    int num = this->prob_[t]->num();
    int dim = this->prob_[t]->count() / num;
    int spatial_dim = prob_[t]->height() * prob_[t]->width();
    CHECK(prob_[t]->height() == 1);
    CHECK(prob_[t]->width() == 1);
    Dtype loss = 0;
    int sample_cnt_ = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; j++) {
        int raw_label = static_cast<int>(label[i * spatial_dim + j]);
        int this_label = new_labels_[t][raw_label];
        sample_cnt_++;
        if (this_label >= 0) {
          loss -= log(std::max(prob_data[i * dim +
                                this_label * spatial_dim + j],
                               Dtype(FLT_MIN)));
        }
      }
    }
    if (sample_cnt_ > 0)
      node_loss_[t] = loss / num / spatial_dim;
    else
      ; //(*top)[0]->mutable_cpu_data()[0] = 0;
  }
  
  // output the loss per tree layer
  for (int d = 0; d < tree_depth_; ++d) {
    (*top)[d]->mutable_cpu_data()[0] = 0;
    for (int n = depth_end_position_[d]; n < depth_end_position_[d+1]; ++n) {
      (*top)[d]->mutable_cpu_data()[0] += node_loss_[n];
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossTreeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[num_nodes_]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    int num = prob_[0]->num();
    // initialize the bottom diff to zeros
    for (int t = 0; t < num_nodes_; ++t) {
      Dtype* bottom_diff = (*bottom)[t]->mutable_cpu_diff();
      caffe_set((*bottom)[t]->count(), Dtype(0), bottom_diff);
    }
    const Dtype* label = (*bottom)[num_nodes_]->cpu_data();
    for (int i = 0; i < num; ++i) {
      int node_idx = 0;
      int depth_level = 0;
      while(depth_level < tree_depth_) {
        Dtype* bottom_diff = (*bottom)[node_idx]->mutable_cpu_diff();
        const Dtype* prob_data = prob_[node_idx]->cpu_data();
        int num = prob_[node_idx]->num();
        int dim = prob_[node_idx]->count() / num;
        int spatial_dim = prob_[node_idx]->height() * prob_[node_idx]->width();
        CHECK(prob_[node_idx]->height() == 1);
        CHECK(prob_[node_idx]->width() == 1);
        caffe_copy(dim, prob_data + i * dim, bottom_diff + i * dim);
  
        int this_label;
        for (int j = 0; j < spatial_dim; ++j) {
          int raw_label = static_cast<int>(label[i * spatial_dim + j]);
          this_label = new_labels_[node_idx][raw_label];
          if (this_label >= 0) {
            bottom_diff[i * dim + this_label
                * spatial_dim + j] -= 1;
          }
          else {
            // if we set the diff to zero, we assume the negatives have no gradient
            LOG(INFO) << "should never reach here";
            caffe_set(dim, Dtype(0), bottom_diff + i * dim);
            // otherwise, we should leave the bottom_diff unchanged
          }
        }
        // check if this layer classifies correctly
        std::vector<std::pair<Dtype, int> > prob_data_vector;
        for (int j = 0; j < dim; ++j) {
          prob_data_vector.push_back(
              std::make_pair(prob_data[i * dim + j], j));
        }
        std::partial_sort(
            prob_data_vector.begin(), prob_data_vector.begin() + 1,
            prob_data_vector.end(), std::greater<std::pair<Dtype, int> >());
        // if not, no gradient from children
        if (prob_data_vector[0].second != this_label) {
          break;
        }
        // if yes, go on with child
        int skip_nodes = 0;
        for (int node = depth_end_position_[depth_level]; node < node_idx; node++) {
          skip_nodes += num_classes_[node];
        }
        node_idx = depth_end_position_[depth_level+1] + skip_nodes + this_label;

        depth_level++;
      }
      // recurssively set the children classifier to be zero
      // ResetChildrenGradient(bottom, i, node_idx, depth_level);
    }
    // Scale gradient
    for (int d = 0; d < tree_depth_; ++d) {
      const Dtype loss_weight = top[d]->cpu_diff()[0];
      for (int n = depth_end_position_[d]; n < depth_end_position_[d+1]; ++n) {
        Dtype* bottom_diff = (*bottom)[n]->mutable_cpu_diff();
        int spatial_dim = prob_[n]->height() * prob_[n]->width();
        caffe_scal(prob_[n]->count(), loss_weight * node_weight_[n] / num / spatial_dim, bottom_diff);
      }
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossTreeLayer<Dtype>::ResetChildrenGradient(vector<Blob<Dtype>*>* bottom, 
    int i, int node_idx, int depth_level) {
  if (depth_level < tree_depth_) {
    int skip_nodes = 0;
    for (int node = depth_end_position_[depth_level]; node < node_idx; node++) {
      skip_nodes += num_classes_[node];
    }
    for (int c = 0; c < num_classes_[node_idx]; ++c) {
      int child_node = depth_end_position_[depth_level+1] + skip_nodes + c;
      Dtype* bottom_diff = (*bottom)[child_node]->mutable_cpu_diff();
      int dim = prob_[child_node]->count() / prob_[child_node]->num();
      caffe_set(dim, Dtype(0), bottom_diff + i * dim);
      ResetChildrenGradient(bottom, i, child_node, depth_level+1);
    }
  }
}

INSTANTIATE_CLASS(SoftmaxWithLossTreeLayer);

}
