# -*- coding: utf-8 -*-
"""
this module uses find_cars function (hog subsampling)
to detect cars in an image

@author: atpandey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time
import pickle
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

from functions import find_cars,draw_boxes


if __name__ == '__main__':
    
    for_pickle=pickle.load(open( 'colorspace_YCrCb_spatialbin_True_colhist_True.p', 'rb' ))
    
    svc=for_pickle['svc']
    X_scaler=for_pickle['scaler']
    color_space=for_pickle['color_space'] 
    spatial_size=for_pickle['spatial_size']
    hist_bins=for_pickle['hist_bins']
    orient=for_pickle['orient']
    pix_per_cell=for_pickle['pix_per_cell']
    cell_per_block=for_pickle['cell_per_block']
    hog_channel=for_pickle['hog_channel']
    spatial_feat=for_pickle['spatial_feat']
    hist_feat=for_pickle['hist_feat']
    hog_feat=for_pickle['hog_feat']
    
    show_scalar=True
    #test_img_bgr = cv2.imread('./test_images/test1.jpg')
    #
    #test_img = convert_color(test_img_bgr,color_space='RGB')
    ffc,axfc=plt.subplots(1,2)
    test_img = mpimg.imread('../test_images/test1.jpg')
    print("img shape",test_img.shape)
    ystart = 400
    ystop = 656
    scale = 1.5

    print("spatial enabled:",spatial_feat,"Hist enabled:",hist_feat)
    bbox = find_cars(test_img, ystart, ystop, scale, color_space, hog_channel, svc, 
                     X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,spatial_feat,hist_feat,axfc=axfc,show_scalar=show_scalar)

    
    print(len(bbox), 'bboxes found in image')
    
    show_scalar=False
    im_bbox=draw_boxes(test_img, bbox, color=(0, 0, 255), thick=6)
    f,p=plt.subplots()
    p.imshow(im_bbox)