# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 16:53:00 2019

@author: DELL
"""

import cv2
import numpy as np
import os

from glob import glob

def main():
    img_mask_left = 'left.jpg'
    img_mask_right = 'right.jpg'
    img_names_left = glob(img_mask_left)
    img_names_right = glob(img_mask_right)

    h, w = cv2.imread(img_names_left[0], cv2.IMREAD_GRAYSCALE).shape[:2]

    #棋盘内角点，每行9个，每列6个
    #pattern_points为自己设置的54*3的矩阵，表示目标点的坐标，z轴坐标恒为0，(x,y)遍历(0,0)到(8,5)
    pattern_size = (9, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    
    #函数输入图片文件名，返回值为内角点的像素坐标，和pattern_points（54*3）矩阵
    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    
        #cv.findChessboardCorners返回值为bool变量和内角点位置
        found, corners = cv2.findChessboardCorners(img, pattern_size)
    
        #增加焦点检测精细度，cv.TERM_CRITERIA_EPS表示角点位置变化的最小值已经达到最小时停止迭代
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            
            if not found:
                print('chessboard not found')
                return None
            
            print('           %s... OK' % fn)
            return (corners.reshape(-1, 2), pattern_points)  
        
    chessboards_left = [processImage(fn) for fn in img_names_left]
    chessboards_left = [x for x in chessboards_left if x is not None]
    
    chessboards_right = [processImage(fn) for fn in img_names_right]
    chessboards_right = [x for x in chessboards_right if x is not None]
    
    #img_points为输入图片的内角点数据，是输入图片数量（这里是13）、内点坐标的三维张量
    #obj_points为相应的，13个pattern_points坐标的3位张量   
    def calibrateCamera(chessboards):
        obj_points = []
        img_points = []
        for (corners, pattern_points) in chessboards:
            img_points.append(corners)
            obj_points.append(pattern_points)
        
        # calculate camera distortion
        rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)       
        #由旋转向量推出旋转矩阵
        R = [cv2.Rodrigues(_rvecs[x])[0] for x in range(np.array(_rvecs).shape[0])]
        newcamera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        return np.array(R), np.array(_tvecs),newcamera_matrix,dist_coefs

    R_left,t_left,cameramtx_left,distCoeffs1 = calibrateCamera(chessboards_left)
    R_right,t_right,cameramtx_right,distCoeffs2 = calibrateCamera(chessboards_right)
    
    #根据公式，计算左右相机变换的R和t
    R = [np.dot(R_right[x],R_left[x].T) for x in range(np.shape(R_left)[0])]
    T = [t_right[x]-np.dot(R[x],t_left[x]) for x in range(np.shape(t_left)[0]) ]
    
    R1 = np.zeros([3,3])
    R2 = np.zeros([3,3])
    P1 = np.zeros([3,4])
    P2 = np.zeros([3,4])
    Q = np.zeros([4,4])   
    R1, R2, P1, P2, Q, Roi1, Roi2=cv2.stereoRectify(cameramtx_left, distCoeffs1,cameramtx_right,distCoeffs2 , (w,h), R[0], T[0])
    
    img1 = cv2.imread('left01.jpg', 0)
    img2 = cv2.imread('right01.jpg', 0)
    img_left = cv2.initUndistortRectifyMap(cameramtx_left, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    img_right = cv2.initUndistortRectifyMap(cameramtx_right, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

    out_left = cv2.remap(img1, img_left[0], img_left[1],cv2.INTER_LINEAR )
    out_right = cv2.remap(img2,img_right[0], img_right[1],cv2.INTER_LINEAR)
    
    
    cv2.imwrite('left_rectifying1.jpg',out_left)
    cv2.imwrite('right_rectifying1.jpg',out_right)   

if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
