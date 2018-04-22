## -----------------------------------------------------------------------------------
## 
##   Copyright (c) 2018 Alex Xiao.  All rights reserved.
## 
##   Description:
##       Implementation of calculating apperance features based on trained GoogLeNet.
## 
## -----------------------------------------------------------------------------------

import numpy as np
import os
import sys
import glob
import caffe
import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog

loc = sys.argv[1]
loc_dir = '../../Track3/' + loc + '/'
vehicle_dir = loc_dir + 'vehicle/'


for time in range(0,2):
  if time == 0:
    inputfile = vehicle_dir + 'left_input.txt'
    outputfile = vehicle_dir + 'left_clr_feat.txt'
    fname = glob.glob(vehicle_dir + 'left/*.jpg')
  else:
    inputfile = vehicle_dir + 'right_input.txt'
    outputfile = vehicle_dir + 'right_clr_feat.txt'
    fname = glob.glob(vehicle_dir + 'right/*.jpg')

  fname.sort(key=lambda x: int(x[-19:-13]))

  with open(inputfile, 'w') as f:
      for name in fname:
          f.write(name + '\n')

  print 'Reading images from "', inputfile
  print 'Writing vectors to "', outputfile

  with open(inputfile, 'r') as reader:
      with open(outputfile, 'w') as writer:
          writer.truncate()
          for image_path in reader:
              print image_path.rstrip('\n')
              image_path = image_path.rstrip()
              image = cv2.imread(image_path)

              mask = np.zeros(image.shape[:2], np.uint8)
              mask = cv2.ellipse(mask, (image.shape[1] / 2,image.shape[0] / 2), (image.shape[1] / 2,image.shape[0] / 2), 0, 0, 360, 255, -1)
              # masked_img = cv2.bitwise_and(image,image, mask=mask)
              # plt.imshow(masked_img, 'gray')
              # plt.show()

              hist1 = cv2.calcHist([image], [0], mask, [16], [0, 256]).reshape(1, -1)
              hist2 = cv2.calcHist([image], [1], mask, [16], [0, 256]).reshape(1, -1)
              hist3 = cv2.calcHist([image], [2], mask, [16], [0, 256]).reshape(1, -1)
              # cv2.normalize(hist1, hist1)
              # cv2.normalize(hist2, hist2)
              # cv2.normalize(hist3, hist3)
              rgb_feat = np.concatenate((hist1, hist2, hist3), axis=1)
              cv2.normalize(rgb_feat, rgb_feat)

              img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
              hist1 = cv2.calcHist([img_hsv], [0], mask, [16], [0, 256]).reshape(1, -1)
              hist2 = cv2.calcHist([img_hsv], [1], mask, [16], [0, 256]).reshape(1, -1)
              # hist3 = cv2.calcHist([img_hsv], [2], mask, [8], [0, 256]).reshape(1, -1)
              # cv2.normalize(hist1, hist1)
              # cv2.normalize(hist2, hist2)
              # cv2.normalize(hist3, hist3)
              hsv_feat = np.concatenate((hist1, hist2), axis=1)
              cv2.normalize(hsv_feat, hsv_feat)

              img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
              # hist1 = cv2.calcHist([img_YCrCb], [0], mask, [8], [0, 256]).reshape(1, -1)
              hist2 = cv2.calcHist([img_YCrCb], [1], mask, [16], [0, 256]).reshape(1, -1)
              hist3 = cv2.calcHist([img_YCrCb], [2], mask, [16], [0, 256]).reshape(1, -1)
              # cv2.normalize(hist1, hist1)
              # cv2.normalize(hist2, hist2)
              # cv2.normalize(hist3, hist3)
              YCrCb_feat = np.concatenate((hist2, hist3), axis=1)
              cv2.normalize(YCrCb_feat, YCrCb_feat)

              img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
              # hist1 = cv2.calcHist([img_lab], [0], mask, [8], [0, 256]).reshape(1, -1)
              hist2 = cv2.calcHist([img_lab], [1], mask, [16], [0, 256]).reshape(1, -1)
              hist3 = cv2.calcHist([img_lab], [2], mask, [16], [0, 256]).reshape(1, -1)
              # cv2.normalize(hist1, hist1)
              # cv2.normalize(hist2, hist2)
              # cv2.normalize(hist3, hist3)
              lab_feat = np.concatenate((hist2, hist3), axis=1)
              cv2.normalize(lab_feat, lab_feat)

              # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
              # image_gray = cv2.resize(image_gray, (200,200))
              # hog_feat = hog(image_gray, orientations=8, pixels_per_cell=(50,50), cells_per_block=(1,1), visualise=False).reshape(1, -1)
              # cv2.normalize(hog_feat, hog_feat)

              type_feat = np.zeros(8).reshape(1,8) + 0.5
              type_feat[0, int(image_path[-5])] = 1
              cv2.normalize(type_feat, type_feat)

              feat = np.concatenate((3 * rgb_feat, hsv_feat, YCrCb_feat, lab_feat, type_feat), axis=1)
              np.savetxt(writer, feat, fmt='%.8g')

