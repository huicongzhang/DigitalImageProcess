import numpy as np
import cmath
import cv2
import time
from BMPread import Bimap
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Gussian import rgb2gray,showimage,PepperandSalt,GaussianNoise,means_Blur,med_Blur
from DWT import haar_wt,haar_iwt,denosie_DWT
from mysobel import get_XY_Sobel

if __name__ == "__main__":
    bitmap = Bimap('/home/zhc/CODE/DigitalImageProcess/EXP3/lena.bmp')
    img = bitmap.rgb
    RGB_img = bitmap.bfimage
    img_gray = rgb2gray(RGB_img)
    GNoiseimg = GaussianNoise(img_gray,0,1,0.2)
    SNoiseImg = PepperandSalt(img_gray,0.2)
    # G_means = means_Blur(SNoiseImg,5)
    # G_med = med_Blur(SNoiseImg,5)
    # showimage([GNoiseimg,Dwt,de_dwt],multi=1,titles=['GNoiseimg','DWT','IDWT'],cmap='gray')
    # G_means = means_Blur(SNoiseImg,5)
    # G_med = med_Blur(SNoiseImg,5)
    # SNoiseImg = np.array(SNoiseImg,dtype=int)
    # Dwt = haar_wt(SNoiseImg)
    # de_dwt = denosie_DWT(Dwt,noise_sigma=8)
    sobel_x,sobel_y,sobel_img  = get_XY_Sobel(img_gray)
    showimage([sobel_img,sobel_x,sobel_y],multi=1,titles=['sobel_img','sobel_x','sobel_y'],cmap='gray')
    