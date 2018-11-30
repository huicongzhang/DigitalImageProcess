#-*- coding: utf-8 -*
#authour Du Changhui

import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt

class BITMAPFILEHEADER(object) :
    # 初始化bmp文件头
    def __init__(self, bfType, bfSize, bf0ffBits ):
        self.bfType = bfType
        self.bfSize = bfSize
        bfReserved1 = 0
        bfReserved2 = 0
        self.bf0ffBits = bf0ffBits

class BITMAPINFOHEADER(object) :
    #初始化位图信息头
    def __init__(self, biSize, biWidth, biHeight, biPlanes, biBitcount, biCompression, biSizeImage, biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant ):
        self.biSize = biSize
        self.biWidth = biWidth
        self.biHeight = biHeight
        self.biPlanes = biPlanes
        self.biBitcount = biBitcount
        self.biCompression = biCompression
        self.biSizeImage = biSizeImage
        self.biXPelsPerMeter = biXPelsPerMeter
        self.biYPelsPerMeter = biYPelsPerMeter
        self.biClrUsed = biClrUsed
        self.biClrImportant = biClrImportant

class tagList(object) :
    #设置调色板信息
    plat = [ ]

class myBmp(object) :
     #读取图片信息
     def __init__(self, pic_path):
         with open(pic_path, 'rb') as file :
             tmp = file.read(14)
             transTmp = struct.unpack('<2cI2HI', tmp)
             self.tagBITMAPFILEHEADER = BITMAPFILEHEADER(transTmp[0]+transTmp[1], transTmp[2], transTmp[5] )

             tmp = file.read(40)
             transTmp = struct.unpack('<3I2H6I',tmp)
             self.tagBITMAPINFOHEADER = BITMAPINFOHEADER(transTmp[0], transTmp[1], transTmp[2], transTmp[3], transTmp[4],
                                                         transTmp[5], transTmp[6], transTmp[7], transTmp[8], transTmp[9],
                                                         transTmp[10])

             #判断有无调色板，并且设定值
             lengthOfPlat = self.tagBITMAPFILEHEADER.bf0ffBits - 54
             if  lengthOfPlat > 0 :
                 colorPlat = tagList()
                 tmp = file.read(lengthOfPlat)
                 length = len(tmp)/4
                 for i in range(length) :
                     rgbBlue = struct.unpack('<c', tmp[0+i*4])
                     rgbGreen = struct.unpack('<c', tmp[1+i*4])
                     rgbRed = struct.unpack('<c', tmp[2+i*4])
                     rgbReserved = struct.unpack('<c', tmp[3+i*4])
                     t = [rgbBlue, rgbGreen, rgbRed, rgbReserved]
                     colorPlat.plat.append(t)
                 RowSize = self.tagBITMAPINFOHEADER.biBitcount * self.tagBITMAPINFOHEADER.biWidth / 8

             else :
                 RowSize =  self.tagBITMAPINFOHEADER.biBitcount * self.tagBITMAPINFOHEADER.biWidth / 8
                 self.bfimage = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
                 for i in range(self.tagBITMAPINFOHEADER.biHeight):
                     s = file.read(int(RowSize))
                     for j in range(self.tagBITMAPINFOHEADER.biWidth):
                         self.bfimage[self.tagBITMAPINFOHEADER.biHeight - i - 1][j][0] = s[j * 3 + 2]
                         self.bfimage[self.tagBITMAPINFOHEADER.biHeight - i - 1][j][1] = s[j * 3 + 1]
                         self.bfimage[self.tagBITMAPINFOHEADER.biHeight - i - 1][j][2] = s[j * 3]
                 self.rgb = self.bfimage/255

     def image(self):
         image1 = self.bfimage
         return image1
     def R_G_B(self):
         R = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
         G = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
         B = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
         R[:,:,2], G[:,:,1], B[:,:,0] = self.bfimage[:,:,0], self.bfimage[:,:,1], self.bfimage[:,:,2]
         return R, G, B

     def Y_I_Q(self):
         Transform_matrix = [
             [0.299, 0.587, 0.114],
             [0.596, -0.274, -0.322],
             [0.211, -0.523, 0.312]
         ]
         self.YIQ = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
         for i in range(self.tagBITMAPINFOHEADER.biHeight):
             for j in range(self.tagBITMAPINFOHEADER.biWidth):
                 self.YIQ[i][j] = np.dot(Transform_matrix, self.bfimage[i][j])

         return self.YIQ

     def H_S_I(self):
         self.HSI = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))

         self.HSI[:, :, 2] = np.sum(self.rgb, axis=2) / 3
         rgbmin = np.min(self.rgb, axis=2)
         rgbsum = np.sum(self.rgb, axis=2)
         den = np.sqrt((self.rgb[:, :, 0] - self.rgb[:, :, 1]) ** 2 + (self.rgb[:, :, 0] - self.rgb[:, :, 2]) * (
                     self.rgb[:, :, 1] - self.rgb[:, :, 2]))
         dd = 0.5 * (2 * self.rgb[:, :, 0] - self.rgb[:, :, 1] - self.rgb[:, :, 2])

         for i in range(self.tagBITMAPINFOHEADER.biHeight):
             for j in range(self.tagBITMAPINFOHEADER.biWidth):
                 if den[i, j] <= 0:
                     self.HSI[i, j, 0] = 0
                 elif den[i, j] > 0 and self.rgb[i, j, 1] >= self.rgb[i, j, 2]:
                     self.HSI[i, j, 0] = np.arccos(int(dd[i, j] / den[i, j]))
                 elif den[i, j] > 0 and self.rgb[i, j, 1] < self.rgb[i, j, 2]:
                     self.HSI[i, j, 0] = 2 * np.pi - np.arccos(int(dd[i, j] / den[i, j]))
         for i in range(self.tagBITMAPINFOHEADER.biHeight):
             for j in range(self.tagBITMAPINFOHEADER.biWidth):
                 if self.HSI[i, j, 2] == 0:
                     self.HSI[i, j, 1] = 0
                 else:
                     self.HSI[i, j, 1] = 1 - (3 * rgbmin[i, j] / rgbsum[i, j])
         return self.HSI

     def X_Y_Z(self):
         Transform_matrix = [
             [0.490, 0.310, 0.200],
             [0.177, 0.813, 0.011],
             [0., 0.010, 0.990]
         ]
         self.XYZ = np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
         for i in range(self.tagBITMAPINFOHEADER.biHeight):
             for j in range(self.tagBITMAPINFOHEADER.biWidth):
                 self.XYZ[i][j] = np.dot(Transform_matrix, self.rgb[i][j])
         return self.XYZ

     def Y_Cb_Cr(self):
         Transform_matrix = [
             [0.299, 0.587, 0.114],
             [0.500, -0.4187, -0.0813],
             [-0.1687, -0.3313, 0.500]
         ]
         self.Y_CbCr=np.zeros(shape=(self.tagBITMAPINFOHEADER.biHeight, self.tagBITMAPINFOHEADER.biWidth, 3))
         for i in range(self.tagBITMAPINFOHEADER.biHeight):
             for j in range(self.tagBITMAPINFOHEADER.biWidth):
                 self.Y_CbCr[i][j] = np.dot(Transform_matrix, self.rgb[i][j])
         return self.Y_CbCr

if __name__ == "__main__":
    bitmap = myBmp(r'C:\Users\xiao\Desktop\LENA.BMP')
    print("请输入要显示的颜色空间："
          "1.RGB "
          "2.YIQ "
          "3.HSI "
          "4.XYZ "
          "5.YCbCr")
    num = input()

    if num == '1' :
        B,G,R = bitmap.R_G_B()
        cv2.imshow('G', G)
        cv2.imshow('R', R)
        cv2.imshow('B', B)
        cv2.waitKey()
    elif num == '2' :
        Y,I,Q = bitmap.Y_I_Q()[:, :, 0], bitmap.Y_I_Q()[:, :, 1], bitmap.Y_I_Q()[:, :, 2]
        cv2.imshow('Y', Y)
        cv2.imshow('I', I)
        cv2.imshow('Q', Q)
        cv2.waitKey()
    elif num == '3' :
        HSI = bitmap.H_S_I()
        H = HSI[:, :, 0]
        H = H / (2 * np.pi)
        I = HSI[:, :, 2]
        S = HSI[:, :, 1]
        cv2.imshow('X', H)
        cv2.imshow('Y', I)
        cv2.imshow('Z', S)
        cv2.waitKey()
    elif num == '4' :
        X, Y, Z = bitmap.X_Y_Z()[:, :, 0], bitmap.X_Y_Z()[:, :, 1], bitmap.X_Y_Z()[:, :, 2]
        cv2.imshow('X', X)
        cv2.imshow('Y', Y)
        cv2.imshow('Z', Z)
        cv2.waitKey()
    elif num == '5' :
        Y, Cb, Cr = bitmap.Y_Cb_Cr()[:, :, 0], bitmap.Y_Cb_Cr()[:, :, 1], bitmap.Y_Cb_Cr()[:, :, 2]
        cv2.imshow('Y', Y)
        cv2.imshow('Cb', Cb)
        cv2.imshow('Cr', Cr)
        cv2.waitKey()




















