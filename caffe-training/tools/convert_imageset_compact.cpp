// Copyright 2014 BVLC and contributors.
// This program converts a set of images to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4 || argc > 5) {
    printf("Convert a set of images to the leveldb format (JPEG-compressed) used\n"
        "as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME"
        " RANDOM_SHUFFLE_DATA[0 or 1]\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
    return 1;
  }
  std::ifstream infile(argv[2]);
  std::vector<std::pair<string, int> > lines;
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines.push_back(std::make_pair(filename, label));
  }
  if (argc == 5 && argv[4][0] == '1') {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

  string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  unsigned char buf[1000000];  // buffer for JPEG data
  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    string img_path = root_folder + lines[line_id].first;
    int label = lines[line_id].second;
    if (line_id < 20)
      LOG(INFO) << img_path << ", " << label;
    FILE *fp = fopen(img_path.c_str(), "rb");
    if (!fp) {
      LOG(INFO) << img_path << " failed";
      continue;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp); // get the size of the JPEG data
    fseek(fp, 0, SEEK_SET); // move fp to the beginning
    int *p = (int *)buf;
    *p = label;
    fread(buf + sizeof(int), sizeof(char), size, fp); // read JPEG data
    fclose(fp);
    // if (!ReadImageToDatum(root_folder + lines[line_id].first,
    //                       lines[line_id].second, &datum)) {
    //   continue;
    // }
    // if (!data_size_initialized) {
    //   data_size = datum.channels() * datum.height() * datum.width();
    //   data_size_initialized = true;
    // } else {
    //   const string& data = datum.data();
    //   CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
    //       << data.size();
    // }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].first.c_str());
    string value(buf, buf + sizeof(int) + size); // initilize string from buf
    // get the value
    // datum.SerializeToString(&value);
    batch->Put(string(key_cstr), value);
    if (++count % 1000 == 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Processed " << count << " files.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR) << "Processed " << count << " files.";
  }

  delete batch;
  delete db;
  return 0;
}
