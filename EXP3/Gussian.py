import numpy as np
import cmath
import cv2
import time
from BMPread import Bimap
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def GaussianNoise(src,means,sigma,noise_scale):
    NoiseImg=src.copy()
    rows=NoiseImg.shape[0]
    cols=NoiseImg.shape[1]
    noise_num = int(rows*cols*noise_scale)
    while noise_num >= 0:
        i = np.random.randint(0,rows-1)
        j = np.random.randint(0,rows-1)
        noise = random.gauss(means,sigma)
        NoiseImg[i,j]=NoiseImg[i,j]+random.gauss(means,sigma)
        for k in range(3):
            if  NoiseImg[i,j,k]< 0:
                NoiseImg[i,j,k]=0
            elif  NoiseImg[i,j,k]>1:
                NoiseImg[i,j,k]=1
        noise_num -= 1
    return NoiseImg
def PepperandSalt(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.randint(0,src.shape[0]-1)
        randY=np.random.randint(0,src.shape[1]-1)
        if np.random.randint(0,1)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=1          
    return NoiseImg
def showimage(imgs,multi=None,titles='origin',cmap=None):
    plt.figure(1)
    if multi == None:
        plt.title(titles)
        plt.imshow(imgs)
    else:
            
            for i in range(len(titles)):
                plt.subplot2grid((1,len(titles)), (0,i))
                plt.title(titles[i])
                plt.imshow(imgs[i],cmap=cmap) 
            """
            plt.subplot2grid((2,3), (1,1))
            plt.title(titles[2])
            plt.imshow(imgs[2],cmap=cmap) """
    plt.show()
def means_Blur(src,kernel_size):
    src_cpy = src.copy()
    padding = int((kernel_size-1)/2)
    means_kernel = np.ones((kernel_size,kernel_size))/9
    rows = src.shape[0]+padding
    cols = src.shape[1]+padding
    img = np.zeros((rows,cols,3))
    img[padding:rows,padding:cols] = src
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            ri = i+padding
            rj = j+padding
            for k in range(3):
                img[ri-padding:ri+padding,rj-padding:rj+padding,k]*means_kernel
                # src_cpy[i][j][k] = np.sum(np.sum(img[ri-padding:ri+padding,rj-padding:rj+padding,k]*means_kernel,axis=0),axis=1)
    return src_cpy
if __name__ == "__main__":
    bitmap = Bimap('/Users/zhanghuicong/CODE/DigitalImageProcess/EXP3/lena.bmp')
    img = bitmap.rgb
    # GNoiseImg = GaussianNoise(img,0,0.1,0.2)
    # SNoiseImg = PepperandSalt(img,0.2)
    img = means_Blur(img,3)
    plt.title('sa')
    plt.imshow(img)
    plt.show()
    # showimage([img,GNoiseImg,SNoiseImg],multi=1,titles=['source','Gaussian','PepperandSalt'])
    
