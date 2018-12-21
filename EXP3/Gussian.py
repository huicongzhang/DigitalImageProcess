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
        NoiseImg[i,j]=NoiseImg[i,j]+random.gauss(means,sigma)*64
        
        if  NoiseImg[i,j]< 0:
            NoiseImg[i,j]=0
        elif  NoiseImg[i,j]>255:
            NoiseImg[i,j]=255
        noise_num -= 1
    return NoiseImg
def PepperandSalt(src,percetage):
    NoiseImg=src.copy()
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=np.random.randint(0,src.shape[0]-1)
        randY=np.random.randint(0,src.shape[1]-1)
        if np.random.rand() < 0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255      
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
    means_kernel = np.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)
    rows = src.shape[0]+padding
    cols = src.shape[1]+padding
    img = np.zeros((rows,cols))
    img[padding:rows,padding:cols] = src
    print(img.shape)
    i = 0
    j = 0
    for ri in range(img.shape[0]-kernel_size):
        j = 0
        for rj in range(img.shape[1]-kernel_size):
            
                # iii = img[ri-padding:ri+padding+1,rj-padding:rj+padding+1,k]*means_kernel
                # iii = np.sum(img[ri-padding:ri+padding+1,rj-padding:rj+padding+1,k]*means_kernel,axis=0)
                # print(iii.shape)
                
                # print(img[ri:ri+kernel_size,rj:rj+kernel_size,k].shape)
                # print(rj+padding+1)
                
            src_cpy[i][j] = np.sum(np.sum(img[ri:ri+kernel_size,rj:rj+kernel_size]*means_kernel,axis=0),axis=0)
            # print(j)
            j += 1
            # print(rj)
        i += 1
    return src_cpy
""" def get_H(img):
    # th = int(img.shape[0]*img.shape[1]/2) + 1
    H = np.zeros((img.shape[2],256))
    # med = np.zeros((img.shape[2]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                H[k,img[i][j][k]] += 1
    
    return H """
""" def get_med(H,img):
    med = np.zeros((img.shape[2]))
    th = int(img.shape[0]*img.shape[1]/2) + 1
    count_flag = 0
    # index_flag = 0
    
    for k in range(img.shape[2]):
        count_flag = 0
        for i in range(256):
            count_flag+=H[k,i]
            if count_flag == th:
                med[k] = i
    return med,th """
def get_H(img):
    # th = int(img.shape[0]*img.shape[1]/2) + 1
    H = np.zeros((256))
    # med = np.zeros((img.shape[2]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            H[int(img[i][j])] += 1
    return H
def get_med(H,img):
    med = 0
    th = int(img.shape[0]*img.shape[1]/2) + 1
    count_flag = 0
    # index_flag = 0
    for i in range(256):
        count_flag+=H[i]
        if count_flag >= th:
            return i,th
""" def med_Blur(src,kernel_size):
    src_cpy = src.copy()
    padding = int((kernel_size-1)/2)
    means_kernel = np.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)
    rows = src.shape[0]+padding
    cols = src.shape[1]+padding
    img = np.zeros((rows,cols))
    img[padding:rows,padding:cols] = src
    print(img.shape)
    
    i = 0
    j = 0
    for ri in range(img.shape[0]-kernel_size):
        med_win = img[ri:ri+kernel_size,0:kernel_size]
        # print("new rows")
        # print(med_win)
        H = get_H(med_win)
        # print(H)
        med,th = get_med(H,med_win)
        mNum = th
        src_cpy[i][j] = med
        # print(med)
        j = 1
        
        for rj in range(1,img.shape[1]-kernel_size):
            print(H)
            L_med_win = med_win[:,0]
            for z in range(kernel_size):
                H[int(L_med_win[z])] -= 1
                if L_med_win[z] <= med:
                    mNum -= 1
            med_win = img[ri:ri+kernel_size,rj:rj+kernel_size]
            # print(med_win[:,kernel_size-1])
            R_med_win = med_win[:,kernel_size-1]
            # print(R_med_win)
            for z in range(kernel_size):
                H[int(R_med_win[z])] += 1
                if R_med_win[z] <= med:
                    mNum += 1
            if mNum > th:
                while mNum > th: 
                    mNum -= H[med]
                    med -= 1
            else:
                 while mNum < th:
                    print(med)
                    med += 1
                    mNum += H[med]
            src_cpy[i][j] = med
            j += 1
        i += 1
            # src_cpy[i][j][k] = img[ri:ri+kernel_size,rj:rj+kernel_size,k]
    return src_cpy """
def med_Blur(src,kernel_size):
    src_cpy = src.copy()
    padding = int((kernel_size-1)/2)
    means_kernel = np.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)
    rows = src.shape[0]+padding
    cols = src.shape[1]+padding
    img = np.zeros((rows,cols))
    img[padding:rows,padding:cols] = src
    # print(img.shape)
    
    i = 0
    j = 0
    for ri in range(img.shape[0]-kernel_size):
        j = 0
        for rj in range(img.shape[1]-kernel_size):
            med_win = img[ri:ri+kernel_size,rj:rj+kernel_size]
            H = get_H(med_win)
            med,th = get_med(H,med_win)
            src_cpy[i][j] = med
            j += 1
        i += 1
    return src_cpy
def rgb2gray(image):

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    y = np.zeros_like(image).astype(np.uint8)
    y = R * 0.299 + G * 0.587 + B * 0.114
    return y
if __name__ == "__main__": 
    bitmap = Bimap('/home/zhc/CODE/DigitalImageProcess/EXP3/lena.bmp')
    img = bitmap.rgb
    RGB_img = bitmap.bfimage
    img_gray = rgb2gray(img)
    # print(img_gray.shape)
    """ GNoiseImg = GaussianNoise(img,0,0.1,0.2)
    SNoiseImg = PepperandSalt(img,0.2)
    img = means_Blur(SNoiseImg,5)
    plt.title('sa')
    plt.imshow(img)
    plt.show() """
    """ SNoiseImg = PepperandSalt(img_gray,0.2)
    SNoiseImg = np.array(SNoiseImg*255,dtype=int)
    clear_img = med_Blur(SNoiseImg,5) """
    """ img = [[
        [1,1,1],[2,2,2],[3,3,3]
    ],[
        [4,4,4],[5,5,5],[6,6,6]
    ],[
        [7,7,7],[8,8,8],[9,9,9]
    ]]
    img = np.array(img)
    img = np.reshape(img,(3,3,3))
    print(img[:,:,0])
    print(img.shape)
    H = get_H(img)
    print(H)
    med = get_med(H,img)
    print(med) """
    # showimage([img_gray,SNoiseImg,clear_img],multi=1,titles=['source','PepperandSalt','clear_img'],cmap='gray')
    
