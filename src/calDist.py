## -----------------------------------------------------------------------------------
## 
##   Copyright (c) 2018 Alex Xiao.  All rights reserved.
## 
##   Description:
##       Implementation of calculate top-N candidate in gallery for one vehicle in probe.
## 
## -----------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import cv2
import os
from numpy.matlib import repmat
import shutil
from pyemd import emd_samples
import sys
import math


# unit: s
syncFr11_21 = 710
syncFr12_22 = -339
syncFr13_23 = -1111
syncFr13_24 = 3447
syncFr14_25 = 1757
syncFr14_26 = -438
syncFr31_41 = 965 + 60
syncFr32_42 = 1692


syncFr21_11 = -710
syncFr22_12 = 339
syncFr23_13 = 1111
syncFr24_13 = -3447
syncFr25_14 = -1757
syncFr26_14 = 438
syncFr41_31 = -965 - 60
syncFr42_32 = -1692

loc1 = sys.argv[1]
loc2 = sys.argv[2]
resTxt = 'res.txt'

loc1_dir = '../../Track3/' + loc1 + '/'
loc2_dir = '../../Track3/' + loc2 + '/'


def calSim(p_feat, g_feat):
    d, pn = p_feat.shape
    d, qn = g_feat.shape

    pmag = np.sum(np.multiply(p_feat, p_feat), axis=0)
    qmag = np.sum(np.multiply(g_feat, g_feat), axis=0)

    pmag.shape = (pmag.shape[0], 1)
    m = repmat(qmag, pn, 1) + repmat(pmag, 1, qn) - 2 * np.dot(np.transpose(p_feat), g_feat)

    return m

def gd(x, m, s):
    left = 1 / (math.sqrt(2 * math.pi) * s)
    right = math.exp(-math.pow(x - m, 2) / (2 * math.pow(s, 2)))
    return 1 - left * right

def bhattacharyya(a, b):
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))


if os.path.exists(loc1_dir + 'vehicle/' + loc2):
    shutil.rmtree(loc1_dir + 'vehicle/' + loc2)
os.makedirs(loc1_dir + 'vehicle/' + loc2 + '/right/right')
os.makedirs(loc1_dir + 'vehicle/' + loc2 + '/right/left')
os.makedirs(loc1_dir + 'vehicle/' + loc2 + '/left/right')
os.makedirs(loc1_dir + 'vehicle/' + loc2 + '/left/left')

for time in range(3):
    if time == 0:
        probe = []
        gallery = []
        with open(loc1_dir + 'vehicle/right_input.txt') as f:
            for line in f.readlines():
                probe.append(line.rstrip('\n'))
        with open(loc2_dir + 'vehicle/left_input.txt') as f:
            for line in f.readlines():
                gallery.append(line.rstrip('\n'))

        p_cnn_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/right_cnn_feat.txt', delimiter=' '))
        g_cnn_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/left_cnn_feat.txt', delimiter=' '))

        p_clr_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/right_clr_feat.txt', delimiter=' '))
        g_clr_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/left_clr_feat.txt', delimiter=' '))

    if time == 1:
        probe = []
        gallery = []
        with open(loc1_dir + 'vehicle/right_input.txt') as f:
            for line in f.readlines():
                probe.append(line.rstrip('\n'))
        with open(loc2_dir + 'vehicle/right_input.txt') as f:
            for line in f.readlines():
                gallery.append(line.rstrip('\n'))

        p_cnn_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/right_cnn_feat.txt', delimiter=' '))
        g_cnn_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/right_cnn_feat.txt', delimiter=' '))

        p_clr_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/right_clr_feat.txt', delimiter=' '))
        g_clr_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/right_clr_feat.txt', delimiter=' '))

    elif time == 2:
        probe = []
        gallery = []
        with open(loc1_dir + 'vehicle/left_input.txt') as f:
            for line in f.readlines():
                probe.append(line.rstrip('\n'))
        with open(loc2_dir + 'vehicle/right_input.txt') as f:
            for line in f.readlines():
                gallery.append(line.rstrip('\n'))

        p_cnn_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/left_cnn_feat.txt', delimiter=' '))
        g_cnn_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/right_cnn_feat.txt', delimiter=' '))

        p_clr_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/left_clr_feat.txt', delimiter=' '))
        g_clr_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/right_clr_feat.txt', delimiter=' '))    

    elif time == 3:
        probe = []
        gallery = []
        with open(loc1_dir + 'vehicle/left_input.txt') as f:
            for line in f.readlines():
                probe.append(line.rstrip('\n'))
        with open(loc2_dir + 'vehicle/left_input.txt') as f:
            for line in f.readlines():
                gallery.append(line.rstrip('\n'))

        p_cnn_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/left_cnn_feat.txt', delimiter=' '))
        g_cnn_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/left_cnn_feat.txt', delimiter=' '))

        p_clr_feat = np.transpose(np.genfromtxt(loc1_dir + 'vehicle/left_clr_feat.txt', delimiter=' '))
        g_clr_feat = np.transpose(np.genfromtxt(loc2_dir + 'vehicle/left_clr_feat.txt', delimiter=' ')) 

    p_num = len(probe) / 3
    g_num = len(gallery) / 3

    cnn_dist = np.zeros((p_num, g_num))
    clr_dist = np.zeros((p_num, g_num))
    for i in range(3):
        for j in range(3):
            cnn_dist = cnn_dist + calSim(p_cnn_feat[:, i:p_num*3:3], g_cnn_feat[:, j:g_num*3:3])
            clr_dist = clr_dist + calSim(p_clr_feat[:, i:p_num*3:3], g_clr_feat[:, j:g_num*3:3])
    cnn_dist = cnn_dist / cnn_dist.max()
    clr_dist = clr_dist / clr_dist.max()

    for i in range(p_num):
        probe_id = probe[3 * i][-19: -13]
        print "Process: ", i, loc1, loc2, probe_id

        if time == 0:
            f = open(loc1_dir + '/vehicle/' + loc2 + '/right/left/' + probe[3 * i][-19: -13] + '.txt', 'w')
        elif time == 1:
            f = open(loc1_dir + '/vehicle/' + loc2 + '/right/right/' + probe[3 * i][-19: -13] + '.txt', 'w')
        elif time == 2:
            f = open(loc1_dir + '/vehicle/' + loc2 + '/left/right/' + probe[3 * i][-19: -13] + '.txt', 'w')
        elif time == 3:
            f = open(loc1_dir + '/vehicle/' + loc2 + '/left/left/' + probe[3 * i][-19: -13] + '.txt', 'w')

        f.write(probe[3 * i] + '\t0\n')
        f.write(probe[3 * i + 1] + '\t0\n')
        f.write(probe[3 * i + 2] + '\t0\n')
        
        res = np.zeros((g_num, 2))

        for j in range(g_num):
            res[j, 0] = j
            res[j, 1] = 1e10
            time_diff = (int(probe[3 * i][-12: -6]) - int(gallery[3 * j][-12: -6])) / 30.0
            
            flag = 1
            sim = 0
            if loc1 == 'Loc1_1' and loc2 == 'Loc2_1':
                if time == 0:
                    flag = time_diff > syncFr11_21 - 240 and time_diff < syncFr11_21 - 150
                    sim = gd(time_diff - syncFr11_21 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr11_21 + 150 and time_diff < syncFr11_21 + 240
                    sim = gd(time_diff - syncFr11_21 - 180, 0, 30)

            if loc1 == 'Loc2_1' and loc2 == 'Loc1_1':
                if time == 0:
                    flag = time_diff > syncFr21_11 - 240 and time_diff < syncFr21_11 - 150
                    sim = gd(time_diff - syncFr11_21 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr21_11 + 150 and time_diff < syncFr21_11 + 240
                    sim = gd(time_diff - syncFr11_21 - 180, 0, 30)

            elif loc1 == 'Loc1_2' and loc2 == 'Loc2_2':
                if time == 0:
                    flag = time_diff > syncFr12_22 - 240 and time_diff < syncFr12_22 - 150
                    sim = gd(time_diff - syncFr12_22 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr12_22 + 450 and time_diff < syncFr12_22 + 900 # a lot of traffic
                    sim = gd(time_diff - syncFr12_22 + 650, 0, 50)

            elif loc1 == 'Loc1_3' and loc2 == 'Loc2_3':
                if time == 0:
                    flag = time_diff > syncFr13_23 - 240 and time_diff < syncFr13_23 - 150
                    sim = gd(time_diff - syncFr13_23 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr13_23 + 150 and time_diff < syncFr13_23 + 240
                    sim = gd(time_diff - syncFr13_23 + 180, 0, 30)

            elif loc1 == 'Loc2_3' and loc2 == 'Loc1_3':
                if time == 0:
                    flag = time_diff > syncFr23_13 - 240 and time_diff < syncFr23_13 - 150
                    sim = gd(time_diff - syncFr13_23 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr23_13 + 150 and time_diff < syncFr23_13 + 240
                    sim = gd(time_diff - syncFr13_23 + 180, 0, 30)

            elif loc1 == 'Loc1_3' and loc2 == 'Loc2_4':
                if time == 0:
                    flag = time_diff > syncFr13_24 - 240 and time_diff < syncFr13_24 - 150
                    sim = gd(time_diff - syncFr13_24 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr13_24 + 150 and time_diff < syncFr13_24 + 240
                    sim = gd(time_diff - syncFr13_24 + 180, 0, 30)

            elif loc1 == 'Loc1_4' and loc2 == 'Loc2_5':
                if time == 0:
                    flag = time_diff > syncFr14_25 - 240 and time_diff < syncFr14_25 - 150
                    sim = gd(time_diff - syncFr14_25 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr14_25 + 150 and time_diff < syncFr14_25 + 240
                    sim = gd(time_diff - syncFr14_25 + 180, 0, 30)
            
            elif loc1 == 'Loc1_4' and loc2 == 'Loc2_6':
                if time == 0:
                    flag = time_diff > syncFr14_26 - 240 and time_diff < syncFr14_26 - 150
                    sim = gd(time_diff - syncFr14_26 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr14_26 + 150 and time_diff < syncFr14_26 + 240
                    sim = gd(time_diff - syncFr14_26 + 180, 0, 30)

            elif loc1 == 'Loc2_6' and loc2 == 'Loc1_4':
                if time == 0:
                    flag = time_diff > syncFr26_14 - 240 and time_diff < syncFr26_14 - 150
                    sim = gd(time_diff - syncFr14_26 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr26_14 + 150 and time_diff < syncFr26_14 + 240
                    sim = gd(time_diff - syncFr14_26 + 180, 0, 30)

            elif loc1 == 'Loc2_5' and loc2 == 'Loc1_4':
                if time == 0:
                    flag = time_diff > syncFr25_14 - 240 and time_diff < syncFr25_14 - 150
                    sim = gd(time_diff - syncFr14_26 + 180, 0, 30)
                else:
                    flag = time_diff > syncFr25_14 + 150 and time_diff < syncFr25_14 + 240
                    sim = gd(time_diff - syncFr14_26 + 180, 0, 30)

            # elif loc1 == 'Loc3_1' and loc2 == 'Loc4_1':
            #     if time == 0:
            #         flag = time_diff > syncFr31_41 - 200 and time_diff < syncFr31_41 - 60
            #         # sim = gd(time_diff - syncFr31_41 + 100, 0, 30)
            #     else:
            #         flag = time_diff > syncFr31_41 + 60 and time_diff < syncFr31_41 + 200
            #         # sim = gd(time_diff - syncFr31_41 + 100, 0, 30)

            # elif loc1 == 'Loc4_1' and loc2 == 'Loc3_1':
            #     if time == 0:
            #         flag = time_diff > syncFr41_31 - 300 and time_diff < syncFr41_31 - 0
            #         # sim = gd(time_diff - syncFr31_41 + 100, 0, 30)
            #     else:
            #         flag = time_diff > syncFr41_31 + 0 and time_diff < syncFr41_31 + 300
            #         # sim = gd(time_diff - syncFr31_41 + 100, 0, 30)

            # elif loc1 == 'Loc3_2' and loc2 == 'Loc4_2':
            #     if time == 0:
            #         flag = time_diff > syncFr32_42 - 120 and time_diff < syncFr32_42 - 60
            #         # sim = gd(time_diff - syncFr32_42 + 180, 0, 30)
            #     else:
            #         flag = time_diff > syncFr32_42 + 60 and time_diff < syncFr32_42 + 120
            #         # sim = gd(time_diff - syncFr32_42 + 100, 0, 30)
            
            if flag:
                cnn_sim = 0
                clr_sim = 0

                for ii in range(3):
                    for jj in range(3):
                        cnn_sim = cnn_sim + bhattacharyya(p_cnn_feat[:, 3*i+ii]/ sum(p_cnn_feat[:, 3*i+ii]), g_cnn_feat[:, 3*j+jj] / sum(g_cnn_feat[:, 3*j+jj]))
                        clr_sim = clr_sim + bhattacharyya(p_clr_feat[:, 3*i+ii]/ sum(p_clr_feat[:, 3*i+ii]), g_clr_feat[:, 3*j+jj] / sum(g_clr_feat[:, 3*j+jj]))
                if clr_sim < 2:
                    res[j,1] = cnn_sim + clr_sim

        res = res[res[:,1].argsort()]

        for j in range(200):
            if res[j,1] < 1e10:
                f.write(gallery[3 * int(res[j,0])] + '\t%0.8g\n' % res[j,1])
                f.write(gallery[3 * int(res[j,0]) + 1] + '\t%0.8g\n' % res[j,1])
                f.write(gallery[3 * int(res[j,0]) + 2] + '\t%0.8g\n' % res[j,1])

        f.close()
