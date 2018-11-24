import  struct
import numpy as np
import cv2
import matplotlib.pyplot as plt
class tagBITMAPFILEHEADER:
    def __init__(self,bfType,bfSize,bfReserved1,bfReserved2,bfOffBits):
        self.bfType = bfType
        self.bfSize = bfSize
        self.bfReserved1 = bfReserved1
        self.bfReserved2 = bfReserved2
        self.bfOffBits = bfOffBits
class tagBITMAPINFOHEADER:
    def __init__(self,biSize,biWidth,biHeight,biPlanes,biBitCount,biCompression,biSizeImage,biXPelsPerMeter,biYPelsPerMeter,biClrUsed,biClrImportant):
        self.biSize = biSize #本结构所占用字节数
        self.biWidth = biWidth #位图的宽度(不包括0填充的像素)
        self.biHeight = biHeight #位图的高度(不包括0填充的像素)
        self.biPlanes = biPlanes #目标设备的级别
        self.biBitCount = biBitCount #每个像素所需的位数
        self.biCompression = biCompression #位图压缩类型
        self.biSizeImage = biSizeImage #位图的大小(包括0填充的像素)
        self.biXPelsPerMeter = biXPelsPerMeter #位图水平分辨率
        self.biYPelsPerMeter = biYPelsPerMeter #位图垂直分辨率
        self.biClrUsed = biClrUsed #位图实际使用的颜色表中的颜色数
        self.biClrImportant = biClrImportant #位图显示过程中重要的颜色数
class Bimap:
    def __init__(self,dir):
        with open(dir, 'rb') as f:
            s = f.read(14)
            bit2str = struct.unpack('<ccIHHI', s)
            self.Bitfileheader = tagBITMAPFILEHEADER(
                bit2str[0]+bit2str[1],
                bit2str[2],
                bit2str[3],
                bit2str[4],
                bit2str[5]
            )
            s = f.read(40)
            bit2str = struct.unpack('<IIIHHIIIIII',s)
            self.Bitmapifoheader = tagBITMAPINFOHEADER(
                biSize = bit2str[0],
                biWidth = bit2str[1],
                biHeight = bit2str[2],
                biPlanes = bit2str[3],
                biBitCount = bit2str[4],
                biCompression = bit2str[5],
                biSizeImage = bit2str[6],
                biXPelsPerMeter = bit2str[7],
                biYPelsPerMeter = bit2str[8],
                biClrUsed = bit2str[9],
                biClrImportant = bit2str[10]
            )
            # print(int(self.Bitfileheader.bfSize) - int(Bitmapifoheader.biSizeImage))
            # print(Bitmapifoheader.biSizeImage)
            RowSize = 4*self.Bitmapifoheader.biBitCount*self.Bitmapifoheader.biWidth/32
            
            # print(s)
            self.bfimage = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3),dtype=np.uint8)
            for i in range(self.Bitmapifoheader.biHeight):
                s = f.read(int(RowSize))
                for j in range(self.Bitmapifoheader.biWidth):
                        self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][0] = s[j*3+2]
                        self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][1] = s[j*3+1]
                        self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][2] = s[j*3]
            self.rgb = self.bfimage/255
    def R_G_B(self):
        R = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3),dtype=np.uint8)
        G = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3),dtype=np.uint8)
        B = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3),dtype=np.uint8)
        R[:,:,0],G[:,:,1],B[:,:,2] = self.bfimage[:,:,0],self.bfimage[:,:,1],self.bfimage[:,:,2]
        return R,G,B
    def Y_I_Q(self):
        T_matrix = [
                    [0.299,0.587,0.114],
                    [0.596,-0.274,-0.322],
                    [0.211,-0.523,0.312]
                    ]
        self.YIQ = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3))
        for i in range(self.Bitmapifoheader.biHeight):
            for j in range(self.Bitmapifoheader.biWidth):
                self.YIQ[i][j] = np.dot(T_matrix,self.bfimage[i][j])
        
        return self.YIQ[:,:,0],self.YIQ[:,:,1],self.YIQ[:,:,2]
    def H_S_I(self):
        self.HSI = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3))
        
        self.HSI[:,:,2] = np.sum(self.rgb,axis=2)/3
        rgbmin = np.min(self.rgb,axis=2)
        rgbsum = np.sum(self.rgb,axis=2)
        den = np.sqrt((self.rgb[:,:,0] - self.rgb[:,:,1])**2 + (self.rgb[:,:,0]-self.rgb[:,:,2])*(self.rgb[:,:,1]-self.rgb[:,:,2]))
        dd = 0.5*(2*self.rgb[:,:,0]-self.rgb[:,:,1]-self.rgb[:,:,2])
        


                
        for i in range(self.Bitmapifoheader.biHeight):
            for j in range(self.Bitmapifoheader.biWidth):
                if den[i,j] <= 0:
                    self.HSI[i,j,0] = 0
                elif den[i,j] > 0 and self.rgb[i,j,1] >= self.rgb[i,j,2]:
                    self.HSI[i,j,0] = np.arccos(int(dd[i,j]/den[i,j]))
                elif den[i,j] > 0 and self.rgb[i,j,1] < self.rgb[i,j,2]:
                    self.HSI[i,j,0] = 2*np.pi - np.arccos(int(dd[i,j]/den[i,j]))
        for i in range(self.Bitmapifoheader.biHeight):
            for j in range(self.Bitmapifoheader.biWidth):
                if self.HSI[i,j,2] == 0:
                    self.HSI[i,j,1] = 0
                else:
                    self.HSI[i,j,1] = 1 - (3*rgbmin[i,j]/rgbsum[i,j])
        return self.HSI
    def X_Y_Z(self):
        T_matrix = [
            [0.490,0.310,0.200],
            [0.177,0.813,0.011],
            [0.,0.010,0.990]
        ]
        self.rgb
        self.XYZ = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3))
        for i in range(self.Bitmapifoheader.biHeight):
            for j in range(self.Bitmapifoheader.biWidth):
                self.XYZ[i][j] = np.dot(T_matrix,self.rgb[i][j])
        return self.XYZ
def showimage(title,img,cmap = None):
    plt.figure(title) # 图像窗口名称
    plt.imshow(img,cmap=cmap)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title(title) # 图像题目
    plt.show()

if __name__ == "__main__":
    
    bitmap = Bimap('/Users/zhanghuicong/CODE/DigitalImageProcess/EXP1/Marshmello.bmp')
    
    HSI= bitmap.H_S_I()
    H = HSI[:,:,0]
    H = H/(2*np.pi)
    I = HSI[:,:,2]
    S = HSI[:,:,1]
    X,Y,Z = bitmap.X_Y_Z()[:,:,0],bitmap.X_Y_Z()[:,:,1],bitmap.X_Y_Z()[:,:,2]
    cv2.imshow('X',X)
    cv2.imshow('Y',Y)
    cv2.imshow('Z',Z)
    cv2.waitKey(0)
