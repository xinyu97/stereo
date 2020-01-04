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
    img_mask = 'left??.jpg'
    img_names = glob(img_mask)

    obj_points = []
    img_points = []
    h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]

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
        
    chessboards = [processImage(fn) for fn in img_names]
    chessboards = [x for x in chessboards if x is not None]

    #img_points为输入图片的内角点数据，是输入图片数量（这里是13）、内点坐标的三维张量
    #obj_points为相应的，13个pattern_points坐标的3位张量
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)
        
    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
            
    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('') 
    
    for fn in img_names :
        name= fn.split('.',1)[0]
        outfile = os.path.join('output',name + '_undistorted.png')

        img = cv2.imread(fn)
        if img is None:
            continue
        
        #调整相机矩阵
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        
        #进行反畸变
        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

    print('Done')       
            
if __name__ == '__main__':
    print(__doc__)
    main()
    cv2.destroyAllWindows()
