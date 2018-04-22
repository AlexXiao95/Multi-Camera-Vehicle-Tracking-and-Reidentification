#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());  
  if (!infile.is_open()) {
    LOG(FATAL) <<"Cannot open file "<<source;
  }
  int label_size = this->layer_param_.label_size();
  //LOG(INFO) << "label size "<<label_size;

  while (true) {
    string filename;
    vector<float> label; // more than one label
    infile >> filename;
    float l;
    if (infile.eof()) {break;}    
    for (int i=0; i<label_size; i++) {
      infile >> l;
      label.push_back(l);
      //LOG(INFO) << l;
    }
    lines_.push_back(std::make_pair(filename, label));    
  }
  LOG(INFO) << "finish reading file ";
  

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    LOG(INFO) << "RAND SEED "<<prefetch_rng_seed;
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();//need a shared random seed in solver file for MPI version.
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                         new_height, new_width, &datum));
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  
  (*top)[1]->Reshape(batch_size, label_size, 1, 1);
  
  this->prefetch_label_.Reshape(batch_size, label_size, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int label_size = this->layer_param_.label_size();
  // datum scales
  const int lines_size = lines_.size();

#ifndef USE_MPI
  for (int item_id = 0; item_id < batch_size; ++item_id) {
#else
  for (int item_id = batch_size * Caffe::mpi_self_rank() * (-1); item_id < batch_size * (Caffe::mpi_all_rank() - Caffe::mpi_self_rank()); ++item_id) {
//      For MPI usage, we collectively read batch_size * all_proc samples. Every process will use its
//      own part of samples. This method is more cache and hard disk efficient compared to dataset splitting.
    bool do_read = (item_id>=0) && (item_id<batch_size);
    if(do_read){
#endif
    // get a blob
    CHECK_GT(lines_size, lines_id_);
    if (!ReadImageToDatum(lines_[lines_id_].first,
          lines_[lines_id_].second,
          new_height, new_width, &datum)) {
      LOG(INFO) << "image does not found: " << lines_[lines_id_].first;
      continue;
    }

    // Apply transformations (mirror, crop...) to the data
    this->data_transformer_.Transform(item_id, datum, this->mean_, top_data);
    for (int l = 0; l < label_size; ++l) {
      //modified for label buffer separation
      top_label[item_id * label_size + l] = datum.label(l); 
      // top_label[l * batch_size + item_id] = datum.label(l);
    }
   // LOG(INFO) << "data point:"<<lines_[lines_id_].first<<" "<<datum.label(0);
#ifdef USE_MPI
    //debugging info
    //LOG(INFO) << "window id processed: "<< windows_id_;
    // std::ofstream fout;
    // char filename[100];
    // sprintf(filename, "window_id_%d.txt", Caffe::mpi_self_rank());
    // fout.open(filename, std::ofstream::out | std::ofstream::app);
    // fout << lines_id_ << " ";
    // fout.close(); 
  }
#endif
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      // if (this->layer_param_.image_data_param().shuffle()) {
      //   ShuffleImages();
      // }    
    }
  }
}

INSTANTIATE_CLASS(ImageDataLayer);

}  // namespace caffe
