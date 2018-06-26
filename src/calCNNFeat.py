## -----------------------------------------------------------------------------------
## 
##   Copyright (c) 2018 Alex Xiao.  All rights reserved.
## 
##   Description:
##       Implementation of calculating CNN features based on trained GoogLeNet.
## 
## -----------------------------------------------------------------------------------


import numpy as np
import os
import sys
import glob
import caffe
import cv2
from matplotlib import pyplot as plt

# Main path to your caffe installation
caffe_root = '/home/kunghung/caffe/'

bvlc = 'bvlc_car'

# Model prototxt file
model_prototxt = caffe_root + 'models/' + bvlc + '/deploy.prototxt'

# Model caffemodel file
model_trained = caffe_root + 'models/' + bvlc + '/bvlc_googlenet.caffemodel'

# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'

# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

# Name of the layer we want to extract
# layer_name = 'pool5/7x7_s1'
layer_name = 'pool5'

sys.path.insert(0, caffe_root + 'python')


loc = sys.argv[1]
loc_dir = '../../Track3/' + loc + '/'
vehicle_dir = loc_dir + 'vehicle/'

for time in range(0,2):
  if time == 0:
    inputfile = vehicle_dir + 'left_input.txt'
    outputfile = vehicle_dir + 'left_cnn_feat.txt'
    outputfile2 = vehicle_dir + 'left_class_feat.txt'
    fname = glob.glob(vehicle_dir + 'left/*.jpg')
  else:
    inputfile = vehicle_dir + 'right_input.txt'
    outputfile = vehicle_dir + 'right_cnn_feat.txt'
    outputfile2 = vehicle_dir + 'right_class_feat.txt'
    fname = glob.glob(vehicle_dir + 'right/*.jpg')

  fname.sort(key=lambda x: int(x[-19:-13]))

  with open(inputfile, 'w') as f:
      for name in fname:
          f.write(name + '\n')

  print 'Reading images from "', inputfile
  print 'Writing vectors to "', outputfile

  caffe.set_mode_gpu()
  net = caffe.Classifier(model_prototxt, model_trained,
                         mean=np.load(mean_path).mean(1).mean(1),
                         channel_swap=(2, 1, 0),
                         raw_scale=255,
                         image_dims=(256, 256))

  # Loading class labels
  with open(imagenet_labels) as f:
      labels = f.readlines()


  file2 = open(outputfile2, 'w')

  with open(inputfile, 'r') as reader:
      with open(outputfile, 'w') as writer:
          writer.truncate()
          for image_path in reader:
              image_path = image_path.strip()
              input_image = caffe.io.load_image(image_path)
              prediction = net.predict([input_image], oversample=False)
              print os.path.basename(image_path), ' : ', labels[prediction[0].argmax()].strip(), ' (', prediction[0][prediction[0].argmax()], ')'
              cnn_feat = net.blobs[layer_name].data[0].reshape(1, -1)
              # cnn_feat = cnn_feat / max(max(cnn_feat))
              np.savetxt(writer, cnn_feat, fmt='%.8g')
              file2.write('%d, %s, %.8f\n' % (prediction[0].argmax(), labels[prediction[0].argmax()].strip(), prediction[0][prediction[0].argmax()]))
