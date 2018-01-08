
"""
This function generates spatial bin based features of an image

@author: atpandey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import bin_spatial,convert_color_bgr

if __name__ == '__main__':
    img_name='../cutouts/cutouts/red_car_for_color_hist.png'
    img = cv2.imread(img_name)
    
    
    rgb=convert_color(img,color_space='RGB')

    hsv=convert_color(img,color_space='HSV')
    
    yuv=convert_color(img,color_space='YUV')
    
    ycrcb=convert_color(img,color_space='YCrCb')
    
    spatial_size=(32, 32)
    feature_vec_r = bin_spatial(rgb[:,:,0],size=spatial_size)
    feature_vec_g = bin_spatial(rgb[:,:,1],size=spatial_size)
    feature_vec_b = bin_spatial(rgb[:,:,2],size=spatial_size)
    
    feature_vec_h = bin_spatial(hsv[:,:,0],size=spatial_size)
    feature_vec_s = bin_spatial(hsv[:,:,1],size=spatial_size)
    feature_vec_vhsv = bin_spatial(hsv[:,:,2],size=spatial_size)
    
    feature_vec_yuv = bin_spatial(yuv[:,:,0],size=spatial_size)
    feature_vec_u = bin_spatial(yuv[:,:,1],size=spatial_size)
    feature_vec_vyuv = bin_spatial(yuv[:,:,2],size=spatial_size)
    
    feature_vec_ycr = bin_spatial(ycrcb[:,:,0],size=spatial_size)
    feature_vec_cr = bin_spatial(ycrcb[:,:,1],size=spatial_size)
    feature_vec_cb = bin_spatial(ycrcb[:,:,2],size=spatial_size)
    
    # Plot features
    plt.imshow(rgb)
    plt.title('Base Image')
    fsp,axsp=plt.subplots(4,3,figsize=(12,8))
    fsp.subplots_adjust(hspace =.8, wspace=.5)
    axsp=axsp.ravel()
    plt.grid('on')
    axsp[0].plot(feature_vec_r)
    axsp[0].set_title('R channel binned Features')
    axsp[1].plot(feature_vec_g)
    axsp[1].set_title('G channel binned Features')
    axsp[2].plot(feature_vec_b)
    axsp[2].set_title('B channel binned Features')
    
    axsp[3].plot(feature_vec_h)
    axsp[3].set_title('H channel binned Features')
    axsp[4].plot(feature_vec_s)
    axsp[4].set_title('S channel binned Features')
    axsp[5].plot(feature_vec_vhsv)
    axsp[5].set_title('VHSV channel binned Features')

    axsp[6].plot(feature_vec_yuv)
    axsp[6].set_title('YYUV channel binned Features')
    axsp[7].plot(feature_vec_u)
    axsp[7].set_title('U channel binned Features')
    axsp[8].plot(feature_vec_vyuv)
    axsp[8].set_title('VYUV channel binned Features')

    axsp[9].plot(feature_vec_ycr)
    axsp[9].set_title('YYCrCb channel binned Features')
    axsp[10].plot(feature_vec_cr)
    axsp[10].set_title('Cr channel binned Features')
    axsp[11].plot(feature_vec_cb)
    axsp[11].set_title('Cb channel binned Features')   
    