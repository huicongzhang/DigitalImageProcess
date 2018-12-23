import numpy as np
import cmath
import cv2
import time
from BMPread import Bimap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def get_F_X_V(img_8):
    F_X_V = np.zeros((8,8),dtype=complex)
    F_U_V = np.zeros((8,8),dtype=complex)
    for i in range(8):
        for v in range(8):
            for k in range(8):
                F_X_V[i][v] += img_8[i][k]*np.exp(complex(0,-1)*2*np.pi*v*k/8)
    for u in range(8):
        for v in range(8):
            for k in range(8):
                F_U_V[u][v] += F_X_V[k][v]*np.exp(complex(0,-1)*2*np.pi*u*k/8)
    return F_U_V
def get_IFF(img_8):
    F_X_V = np.zeros((8,8),dtype=complex)
    F_U_V = np.zeros((8,8),dtype=complex)
    for i in range(8):
        for v in range(8):
            for k in range(8):
                F_X_V[i][v] += img_8[i][k]*np.exp(complex(0,1)*2*np.pi*v*k/8)/8
    for u in range(8):
        for v in range(8):
            for k in range(8):
                F_U_V[u][v] += F_X_V[k][v]*np.exp(complex(0,1)*2*np.pi*u*k/8)/8
    return F_U_V
def showimage(imgs,multi=None,titles='origin',cmap=None):
    plt.figure(1)
    if multi == None:
        plt.title(titles)
        plt.imshow(imgs)
    else:
            plt.subplot2grid((1,2), (0,0))
            plt.title(titles[0])
            plt.imshow(imgs[0],cmap=cmap)
            plt.subplot2grid((1,2), (0,1))
            plt.title(titles[1])
            plt.imshow(imgs[1],cmap=cmap)
            """ for i in range(1,len(titles)):
                plt.subplot2grid((2,len(titles)-1), (1,i-1))
                plt.title(titles[i])
                plt.imshow(imgs[i],cmap=cmap) """
            """ plt.subplot2grid((2,3), (1,1))
            plt.title(titles[2])
            plt.imshow(imgs[2],cmap=cmap) """

            

    plt.show()
def DCTmatric(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    i = (2 * i)+1
    metrix = np.cos(i * j * np.pi / (2 * N)) * np.sqrt(2.0 / N)
    metrix[0, :] = metrix[0, :] / np.sqrt(2)
    return metrix
def get_DCT_kernel(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    i = (2 * i)+1
    metrix = np.cos(i * j * np.pi / (2 * N)) * np.sqrt(2.0 / N)
    metrix[0, :] = metrix[0, :] / np.sqrt(2)
    return metrix
def get_FFT(imgY):
    F_X_V = np.zeros_like(imgY,dtype=complex)
    F_X_V_a = np.zeros_like(imgY)
    F_X_V_b = np.zeros_like(imgY)
    # IFF = np.zeros_like(imgY,dtype=complex)
    DFT_kernel = np.zeros((8,8),dtype=complex)
    # start_time = time.clock()
    for v in range(8):
        for u in range(8):
            DFT_kernel[v][u] = np.exp(complex(0,-1)*2*np.pi*v*u/8)
    for i in range(int(imgY.shape[0]/8)):
        for j in range(int(imgY.shape[1]/8)):
            F_X_V[i*8:(i+1)*8,j*8:(j+1)*8] = np.dot(DFT_kernel.dot(imgY[i*8:(i+1)*8,j*8:(j+1)*8]),DFT_kernel.T)
            # F_X_V[i*8:(i+1)*8,j*8:(j+1)*8] = np.fft.fft(imgY[i*8:(i+1)*8,j*8:(j+1)*8])
            # F_X_V[i*8:(i+1)*8,j*8:(j+1)*8] = get_F_X_V(imgY[i*8:(i+1)*8,j*8:(j+1)*8])
    F_X_V_abs = np.abs(F_X_V)
    # F_X_V_log = np.log(np.abs(F_X_V))
    F_X_V_angle = np.angle(F_X_V)
    # F_X_V_b = F_x_v_angle*180/np.pi
    return F_X_V_abs,F_X_V_angle
def get_IFFT(F):
    IFF = np.zeros_like(imgY,dtype=complex)
    for i in range(int(imgY.shape[0]/8)):
        for j in range(int(imgY.shape[1]/8)):
            IFF[i*8:(i+1)*8,j*8:(j+1)*8] = get_IFF(F[i*8:(i+1)*8,j*8:(j+1)*8])
            # np.fft.ifft(np.fft.ifft(F_X_V_a[i*8:(i+1)*8,j*8:(j+1)*8],axis=0),axis=1)
    return IFF
def get_DCT(img,W_kernel,H_kernel):
    DCT_img = np.zeros_like(img)
    for i in range(int(img.shape[0]/8)):
        for j in range(int(img.shape[1]/8)):
            DCT_img[i*8:(i+1)*8,j*8:(j+1)*8] = np.dot(np.dot(H_kernel,img[i*8:(i+1)*8,j*8:(j+1)*8]), W_kernel.T)
    return DCT_img
def get_IDCT(img,W_kernel,H_kernel,mask_kernel=np.ones((8,8))):
    DCT_img = np.zeros_like(img)
    """ if mask_kernel.all() == None:
        mask_kernel = np.ones_like(img)
    else:
        pass """
    for i in range(int(img.shape[0]/8)):
        for j in range(int(img.shape[1]/8)):
            DCT_img[i*8:(i+1)*8,j*8:(j+1)*8] = np.dot(np.dot(H_kernel.T,img[i*8:(i+1)*8,j*8:(j+1)*8]*mask_kernel), W_kernel)
    return DCT_img
def get_zigzag(K):
    mask_kernel = np.zeros((8,8))
    i,j = 0,0
    up = True
    for t in range(K):
        # line.append(rect[j][i])
        #碰壁后就转向
        mask_kernel[i][j] = 1
        if up:#右上
            if i==K-1:
                j += 1
                up=False#转向
            elif j==0:   
                i += 1
                up=False#转向
            else:
                i += 1
                j -= 1
        else:#左下
            if j==K-1:
                i += 1
                up=True#转向
            elif i==0:
                j += 1
                up=True#转向
            else:
                i -= 1
                j += 1
    return mask_kernel
def get_base_DCT(N,H_kernel,W_kernel):

    a = np.resize(H_kernel,(N*N,1))
    b = np.resize(W_kernel, (1,N * N))
    dct = np.dot(a,b)
    return dct
if __name__ == "__main__":
    # im = cv2.imread('/Users/zhanghuicong/CODE/DigitalImageProcess/EXP2/lena.bmp')
    # imgY = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)[:,:,0]
    bitmap = Bimap('/Users/zhanghuicong/CODE/DigitalImageProcess/EXP2/lena.bmp')
    imgY = bitmap.Y_Cb_Cr()[:,:,0]
    # start_time = time.clock()
    F_X_V_abs,F_X_V_angle = get_FFT(imgY)
    # print(time.clock()-start_time)
    # F_X_V_log = np.log(F_X_V_abs)
    # F_X_V_b = F_x_v_angle*180/np.pi
    # IFFT_abs = get_IFFT(F_X_V_abs)
    
    IFFT_angle = get_IFFT(np.exp(F_X_V_angle*complex(0,1)))
    W_kernel = get_DCT_kernel(8)
    H_kernel = get_DCT_kernel(8)
    DCT_img = get_DCT(imgY,H_kernel,W_kernel)
    # DCT_img = np.log(DCT_img)
    DCT_64_img = get_IDCT(DCT_img,H_kernel,W_kernel)
    # DCT_img = np.log(DCT_img)
    # DCT_K_img = get_zigzag(DCT_img[0:8,0:8],6)
    # print(DCT_K_img)
    # DCT_64_img = np.uint8(DCT_64_img)
    mask_kernel = get_zigzag(32)
    DCT_32_img = get_IDCT(DCT_img,H_kernel,W_kernel,mask_kernel=mask_kernel)
    mask_kernel = get_zigzag(16)
    DCT_16_img = get_IDCT(DCT_img,H_kernel,W_kernel,mask_kernel=mask_kernel)
    mask_kernel = get_zigzag(8)
    DCT_8_img = get_IDCT(DCT_img,H_kernel,W_kernel,mask_kernel=mask_kernel)
    mask_kernel = get_zigzag(4)
    DCT_4_img = get_IDCT(DCT_img,H_kernel,W_kernel,mask_kernel=mask_kernel)
    mask_kernel = get_zigzag(2)
    DCT_2_img = get_IDCT(DCT_img,H_kernel,W_kernel,mask_kernel=mask_kernel)
    mask_kernel = get_zigzag(1)
    DCT_1_img = get_IDCT(DCT_img,H_kernel,W_kernel,mask_kernel=mask_kernel)
    # DCT_32_img = np.uint8(DCT_32_img)
    # end_time = time.clock()
    # use_time = end_time - start_time
    # print(use_time)
    # DCT_img = np.log(DCT_img)
    # DCT_img = np.uint8(DCT_img)
    # showimage([imgY,DCT_img],multi=1,titles=['Y','DCT'],cmap='gray')
    # cv2.imshow('幅度',DCT_img)
    # cv2.waitKey(0)
    # base_DCT = get_base_DCT(8,H_kernel,W_kernel)
    plt.imshow(DCT_1_img,cmap='gray')
    plt.title('base dct')
    plt.show()