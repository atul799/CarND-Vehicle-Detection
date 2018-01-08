
"""
this file tests color_hist function
using cv2 to avoid binning range issue on png image
different color space are tried
@author: atpandey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import color_hist,convert_color_bgr

if __name__ == '__main__':
    img_name='../cutouts/cutouts/red_car_for_color_hist.png'
    img = cv2.imread(img_name)

    rgb=convert_color_bgr(img,color_space='RGB')
    br=(0,256)
    rh, gh, bh, rgbbincen, rgb_feature_vec = color_hist(rgb, nbins=32, bins_range=(br[0], br[1]))
    
    hsv=convert_color(img,color_space='HSV')
    br=(0,256)
    hh, sh, vh, hsvbincen, hsv_feature_vec = color_hist(hsv, nbins=32, bins_range=(br[0], br[1]))
    
    yuv=convert_color(img,color_space='YUV')
    br=(0,256)
    yh, uh, yvh, yuvbincen, yuv_feature_vec = color_hist(yuv, nbins=32, bins_range=(br[0], br[1]))
    
    ycrcb=convert_color(img,color_space='YCrCb')
    br=(0,256)
    ych, crh, cbh, ycrcbbincen, ycrcb_feature_vec = color_hist(ycrcb, nbins=32, bins_range=(br[0], br[1]))
    
    fp,axp=plt.subplots(4,3,figsize=(15,10))
    axp=axp.ravel()
    fp.subplots_adjust(hspace =.8, wspace=.1)
    axp[0].bar(rgbbincen,rh[0])
    plt.xlim(br[0],br[1])
    axp[0].set_title('R Histogram')
    axp[1].bar(rgbbincen,gh[0])
    plt.xlim(br[0],br[1])
    axp[1].set_title('G Histogram')
    axp[2].bar(rgbbincen,bh[0])
    plt.xlim(br[0],br[1])
    axp[2].set_title('B Histogram')
    
    axp[3].bar(hsvbincen,hh[0])
    plt.xlim(br[0],br[1])
    axp[3].set_title('H Histogram')
    axp[4].bar(hsvbincen,sh[0])
    plt.xlim(br[0],br[1])
    axp[4].set_title('S Histogram')
    axp[5].bar(hsvbincen,vh[0])
    plt.xlim(br[0],br[1])
    axp[5].set_title('V Histogram')
    
    axp[6].bar(yuvbincen,yh[0])
    plt.xlim(br[0],br[1])
    axp[6].set_title('Y Histogram')
    axp[7].bar(yuvbincen,uh[0])
    plt.xlim(br[0],br[1])
    axp[7].set_title('U Histogram')
    axp[8].bar(yuvbincen,yvh[0])
    plt.xlim(br[0],br[1])
    axp[8].set_title('YV Histogram')
    
    axp[9].bar(ycrcbbincen,ych[0])
    plt.xlim(br[0],br[1])
    axp[9].set_title('YC Histogram')
    axp[10].bar(ycrcbbincen,crh[0])
    plt.xlim(br[0],br[1])
    axp[10].set_title('Cr Histogram')
    axp[11].bar(ycrcbbincen,cbh[0])
    plt.xlim(br[0],br[1])
    axp[11].set_title('Cb Histogram')
    
    #plt.title("color hist for red_car_for_color_hist")
    