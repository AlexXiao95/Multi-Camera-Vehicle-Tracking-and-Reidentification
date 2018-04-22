#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param)
    : param_(param) {
    phase_ = Caffe::phase();
  }
  virtual ~DataTransformer() {}

  void InitRand();
  void FillInOffsets(int *w, int *h, int width, int height, int crop_size) {
    FillInOffsets(w, h, width, height, crop_size, crop_size);
    // w[0] = 0; h[0] = 0;
    // w[1] = 0; h[1] = height - crop_size;
    // w[2] = width - crop_size; h[2] = 0;
    // w[3] = width - crop_size; h[3] = height - crop_size;
    // w[4] = (width - crop_size) / 2; h[4] = (height - crop_size) / 2;
  }

  void FillInOffsets(int *w, int *h, int width, int height, int crop_w, int crop_h) {
    if (crop_w < width * 2 / 3 && crop_h < height * 2 / 3) {
      // we want to be conservative when the crop is small
      w[0] = 0; h[0] = (height - crop_h) / 2;
      w[1] = width - crop_w; h[1] = (height - crop_h) / 2;
      w[2] = (width - crop_w) / 2; h[2] = 0;
      w[3] = (width - crop_w) / 2; h[3] = height - crop_h;
      w[4] = (width - crop_w) / 2; h[4] = (height - crop_h) / 2;
    }
    else {
      w[0] = 0; h[0] = 0;
      w[1] = 0; h[1] = height - crop_h;
      w[2] = width - crop_w; h[2] = 0;
      w[3] = width - crop_w; h[3] = height - crop_h;
      w[4] = (width - crop_w) / 2; h[4] = (height - crop_h) / 2;
    }
  }

  enum Scaling {SINGLE_SCALE, MULTIPLE_SCALE};
  // prefine 9 ('scale' & 'aspect radio') combinations
  static const int widths_[];
  static const int heights_[];
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param batch_item_id
   *    Datum position within the batch. This is used to compute the
   *    writing position in the top blob's data
   * @param datum
   *    Datum containing the data to be transformed.
   * @param mean
   * @param transformed_data
   *    This is meant to be the top blob's data. The transformed data will be
   *    written at the appropriate place within the blob's data.
   */
  void Transform(const int batch_item_id, const Datum& datum,
                 const Dtype* mean, Dtype* transformed_data);
  void Transform(const int batch_item_id, IplImage *img,
                 const Dtype* mean, Dtype* transformed_data);
  void TransformReturnCoord(const int batch_item_id,
                            IplImage *img, const Dtype* mean,
                            Dtype* transformed_data, float crop_coord[]);

  Caffe::Phase phase_;
 protected:
  virtual unsigned int Rand();
  void TransformSingle(const int batch_item_id, IplImage *img,
               const Dtype* mean, Dtype* transformed_data);
  void TransformSingle(const int batch_item_id, IplImage *img,
               const Dtype* mean, Dtype* transformed_data, float crop_coord[]);
  void TransformMultiple(const int batch_item_id, IplImage *img,
               const Dtype* mean, Dtype* transformed_data);
  // Tranformation parameters
  TransformationParameter param_;
  shared_ptr<Caffe::RNG> rng_;
};



}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

