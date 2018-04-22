#include "caffe/data_layers.hpp"
#include <fstream>

using namespace std;

namespace caffe {

template <typename Dtype>
void LabelTransformLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    LabelTransformParameter trans_param = this->layer_param_.label_transform_param();
    CHECK(trans_param.has_label_transform_file()) << this->layer_param_.name() << " No label transform file provided";
    CHECK_EQ(bottom.size(), 1) << this->layer_param_.name() << "LabelTransformLayer only support one input blob";
    CHECK_EQ(top->size(), 1) << this->layer_param_.name() << "LabelTransformLayer only support one output blob";

    ifstream trans_file(trans_param.label_transform_file().c_str());
    CHECK(trans_file.is_open());
    int label_to, cnt;
    // set<int> from, to;

    // while (trans_file >> label_from >> label_to) {
    //     label_table_.insert(pair<int, int>(label_from, label_to));
    //     from.insert(label_from);
    //     to.insert(label_to);
    // }
    new_labels_.clear();
    cnt = 0;
    while (trans_file >> label_to) {
        new_labels_.push_back(label_to);
        if (label_to >= 0)
            cnt++;
    }

    LOG(INFO) << this->layer_param_.name() << "Transform to "
          << cnt << " labels";
}

template <typename Dtype>
void LabelTransformLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    (*top)[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void LabelTransformLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
    int num = bottom[0]->num();
    const Dtype* label_in = bottom[0]->cpu_data();
    Dtype* label_out = (*top)[0]->mutable_cpu_data();

    for (int i = 0; i < num; i++) {
        label_out[i] = new_labels_[(int)(label_in[i])];
    }
}

template <typename Dtype>
void LabelTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      vector<Blob<Dtype>*>* bottom) {
    if (propagate_down[0]) {
        LOG(FATAL) << this->layer_param_.name()
            << " Layer cannot backpropagate to label inputs.";
    }
}

INSTANTIATE_CLASS(LabelTransformLayer);
}

