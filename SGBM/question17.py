# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:53:13 2020

@author: DELL
"""

import numpy as np
import cv2

imgL = cv2.imread('left.png')
imgR = cv2.imread('right.png')

# disparity range tuning
window_size = 3
min_disp = 0
num_disp = 320 - min_disp

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=3,
    P1=8 * 3 * window_size ** 2,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16
cv2.imwrite('disparity.png',disparity)
