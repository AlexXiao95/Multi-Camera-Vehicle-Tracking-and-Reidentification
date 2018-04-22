## -----------------------------------------------------------------------------------
## 
##   Copyright (c) 2018 Alex Xiao.  All rights reserved.
## 
##   Description:
##       Implementation of extracting 3 frames of each vehicles from tracking result.
## 
## -----------------------------------------------------------------------------------

import pandas as pd
from collections import Counter
import os
import cv2
import shutil
import sys
from matplotlib import pyplot as plt
import multiprocessing
import time

loc = sys.argv[1]   # Location name
mp_flag = int(sys.argv[2])  # Flag for using multiprocessing pool

resTxt = 'res.txt'
loc_dir = '../../Track3/' + loc + '/'

start = time.clock()

# read data
with open(loc_dir + resTxt) as f:
    # read 10000 rows, for debugging
    # data = pd.read_table(f, sep=',', header=None, nrows = 10000, names=['frame', 'id', 'x', 'y', 'w', 'h', 'i', 'j', 'k', 'speed', 'class'])
    data = pd.read_table(f, sep=',', header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'i', 'j', 'k', 'speed', 'class'])

print 'Read file time: ', time.clock() - start

del data['i']
del data['j']
del data['k']

class_dic = {'Hatchback': 0, 'Sedan': 1, 'Pickup': 2, 'Van': 3, "Truck": 4, 'Motorcycle': 5, 'Minibus': 6, 'Bus': 7}

min_frame_num = 50  # ignore vehicle which is tracked less than 50 frames.
min_y = 650 # threshold for y
if loc[3] == '3':
    min_y = 540
elif loc == 'loc4_1':
    min_y = 300
elif loc[3] == '4':
    min_y = 400


# add a column which indicate the area of the vehicle
area = []
for i in range(data.shape[0]):
    area.append(int(data['w'][i] * data['h'][i]))
data['area'] = area
max_id = max(data['id'])

vehicle_dir = loc_dir + 'vehicle/'
if os.path.exists(vehicle_dir):
    shutil.rmtree(vehicle_dir)
os.makedirs(vehicle_dir + 'right/')
os.makedirs(vehicle_dir + 'left/')


def extractCar(args):
    i, data_i = args

    # ignore vehicle which is tracked less than 50 frames.
    if (data_i.shape[0] < min_frame_num or max(data_i['y']) < min_y):
        return
    data_i = data_i.sort_values(by = 'area', ascending = False)
    if data_i['area'].iat[0] < 14400:
        return

    print 'Extract car: ', loc, i
    
    # Determine the vehicle's direction of travel
    if loc == 'Loc4_1':
        cx = data_i['x'].iat[0] + data_i['w'].iat[0] / 2
        cy = data_i['y'].iat[0] + data_i['h'].iat[0] / 2
        direction = cx + 2.6 * cy > 1500
        # img = cv2.imread(loc_dir + 'img1/002691.jpg')
        # cv2.line(img, (0,580), (1500,0), (0,0,0), 2)
        # plt.imshow(img, 'gray')
        # plt.show()
    elif loc[3] == '4':
        cx = data_i['x'].iat[0] + data_i['w'].iat[0] / 2
        cy = data_i['y'].iat[0] + data_i['h'].iat[0] / 2
        direction = cx + 2.6 * cy > 1800
        # img = cv2.imread(loc_dir + 'img1/002691.jpg')
        # cv2.line(img, (0,700), (1800,0), (0,0,0), 2)
        # plt.imshow(img, 'gray')
        # plt.show()  
    else:
        direction = data_i['x'].iat[0] + data_i['w'].iat[0] / 2 > 1920 / 2

    # majority vote of class
    major_class = Counter(data_i['class']).keys()[0]

    # choose frame which has top3 largest area

    choose_fr = data_i[0:15:5]

    for i in range(3):
        fr = int(choose_fr['frame'].iat[i])
        ind = int(choose_fr['id'].iat[i])
        im = cv2.imread(loc_dir + 'img1/%06d.jpg' % fr)
        x1 = int(choose_fr['x'].iat[i])
        y1 = int(choose_fr['y'].iat[i])
        x2 = int(choose_fr['x'].iat[i] + choose_fr['w'].iat[i])
        y2 = int(choose_fr['y'].iat[i] + choose_fr['h'].iat[i])
        new_im = im[y1:y2, x1:x2, :]
        
        if direction:
            img_dir = vehicle_dir + 'right/%06d_%06d_%01d.jpg' % (ind, fr, class_dic[major_class])
            cv2.imwrite(img_dir, new_im)
        else:
            img_dir = vehicle_dir + 'left/%06d_%06d_%01d.jpg' % (ind, fr, class_dic[major_class])
            cv2.imwrite(img_dir, new_im)

if __name__ == '__main__':
    data_list = []
    print "Entering.........."

    if mp_flag:
        print "Using multiprocess.........."
        for i in range(max_id + 1):
            data_list.append(data.loc[data['id'] == i])
        
        num_workers = multiprocessing.cpu_count() * 2 - 1
        pool = multiprocessing.Pool(processes = num_workers)
        
        print 'Using ', num_workers, ' workers.'
        extract_args = zip([i for i in range(max_id + 1)], data_list)
        pool.map(extractCar, extract_args)
    else:
        print "Don't use multiprocess.........."
        for i in range(max_id + 1):
            extractCar((i, data.loc[data['id'] == i]))
    print "Complete extraction."

