import  struct
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
class tagBITMAPFILEHEADER:
    def __init__(self,bfType,bfSize,bfReserved1,bfReserved2,bfOffBits):
        self.bfType = bfType #文件类型 
        self.bfSize = bfSize #位图文件的大小
        self.bfReserved1 = bfReserved1 #位图文件保留字1，必须为0
        self.bfReserved2 = bfReserved2 #位图文件保留字2，也必须是0
        self.bfOffBits = bfOffBits #位图数据的起始位置，文件头的偏移量，单位为字节
class tagBITMAPINFOHEADER:
    def __init__(self,biSize,biWidth,biHeight,biPlanes,
        biBitCount,biCompression,biSizeImage,biXPelsPerMeter,
        biYPelsPerMeter,biClrUsed,biClrImportant):
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
            print ("文件类型:", self.Bitfileheader.bfType)
            print ("位图大小:", self.Bitfileheader.bfSize, "字节")
            print ("偏移量:", self.Bitfileheader.bfOffBits, "字节")

            print ("位图信息头大小:",self.Bitmapifoheader.biSize, "字节")
            print ("宽:", self.Bitmapifoheader.biWidth)
            print ("高:", self.Bitmapifoheader.biHeight)
            print (self.Bitmapifoheader.biBitCount, "位图")
            print ("压缩类型:", self.Bitmapifoheader.biCompression)
            print ("图像大小:", self.Bitmapifoheader.biSizeImage, "字节")
            print ("水平分辨率:", self.Bitmapifoheader.biXPelsPerMeter, "像素/米")
            print ("垂直分辨率:", self.Bitmapifoheader.biYPelsPerMeter, "像素/米")
            print ("使用颜色索引数:", self.Bitmapifoheader.biClrUsed, "(为0的话，则说明使用所有调色板项)")
            print ("有重要影响的颜色索引数:", self.Bitmapifoheader.biClrImportant, "(如果是0，表示都重要)")
            print ("是否有索引表:", "无" if self.Bitfileheader.bfOffBits == 54 else "有")
            print ("索引表大小:", self.Bitfileheader.bfOffBits - 54)
            
            RowSize = 4*self.Bitmapifoheader.biBitCount*self.Bitmapifoheader.biWidth/32
            self.bfimage = np.zeros(shape=(self.Bitmapifoheader.biHeight,self.Bitmapifoheader.biWidth,3),dtype=np.uint8)
            if self.Bitmapifoheader.biBitCount >= 24:
            # print(int(self.Bitfileheader.bfSize) - int(Bitmapifoheader.biSizeImage))
            # print(Bitmapifoheader.biSizeImage)
                
                
                # print(s)
                
                for i in range(self.Bitmapifoheader.biHeight):
                    s = f.read(int(RowSize))
                    for j in range(self.Bitmapifoheader.biWidth):
                            self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][0] = s[j*3+2]
                            self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][1] = s[j*3+1]
                            self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][2] = s[j*3]
                self.rgb = self.bfimage/255
            else:
                self.color_list = np.zeros(shape=((self.Bitfileheader.bfOffBits - 54)/4,4),dtype=np.uint8)
                for i in range((self.Bitfileheader.bfOffBits - 54)/4):
                    s = f.read(4)
                    self.color_list[i][0] = s[0]
                    self.color_list[i][1] = s[1]
                    self.color_list[i][2] = s[2]
                    self.color_list[i][3] = s[3]
                for i in range(self.Bitmapifoheader.biHeight):
                    s = f.read(int(RowSize))
                    for j in range(self.Bitmapifoheader.biWidth):
                            self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][0] = self.color_lists[s[j]][2]
                            self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][1] = self.color_lists[s[j]][1]
                            self.bfimage[self.Bitmapifoheader.biHeight-i-1][j][2] = self.color_lists[s[j]][0]
                self.rgb = self.bfimage/255
            print("(25,25)像素值",self.bfimage[25,25])
            
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
    def Y_Cb_Cr(self):
         Transform_matrix = [
             [0.299, 0.587, 0.114],
             [0.500, -0.4187, -0.0813],
             [-0.1687, -0.3313, 0.500]
         ]
         self.Y_Cb_Cr=np.zeros(shape=(self.Bitmapifoheader.biHeight, self.Bitmapifoheader.biWidth, 3))
         for i in range(self.Bitmapifoheader.biHeight):
             for j in range(self.Bitmapifoheader.biWidth):
                 self.Y_Cb_Cr[i][j] = np.dot(Transform_matrix, self.rgb[i][j])
         return self.Y_Cb_Cr
def showimage(imgs,multi=None,titles='origin',cmap=None):
    plt.figure(1)
    if multi == None:
        plt.title(titles)
        plt.imshow(imgs)
    else:
            plt.subplot2grid((2,3), (0,1))
            plt.title(titles[0])
            plt.imshow(imgs[0])
            plt.subplot2grid((2,3), (1,0))
            plt.title(titles[1])
            plt.imshow(imgs[1],cmap=cmap)

            plt.subplot2grid((2,3), (1,1))
            plt.title(titles[2])
            plt.imshow(imgs[2],cmap=cmap)

            plt.subplot2grid((2,3), (1,2))
            plt.title(titles[3])
            plt.imshow(imgs[3],cmap=cmap)


    plt.show()

if __name__ == "__main__":
    
    bitmap = Bimap('/Users/zhanghuicong/CODE/DigitalImageProcess/EXP1/LENA.BMP')
    Y_Cb_Cr = bitmap.Y_Cb_Cr()
    showimage([bitmap.bfimage,Y_Cb_Cr[:,:,0],Y_Cb_Cr[:,:,1],Y_Cb_Cr[:,:,2]],multi=1,titles=["origin","Y","Cb","Cr"],cmap='gray')
    # XYZ = bitmap.X_Y_Z()
    # showimage([bitmap.bfimage,XYZ[:,:,0],XYZ[:,:,1],XYZ[:,:,2]],multi=1,titles=["origin","X","Y","Z"],cmap='gray')
    # showimage('sda',bitmap.bfimage)
    """ HSI= bitmap.H_S_I()
    H = HSI[:,:,0]
    H = H/(2*np.pi)
    I = HSI[:,:,2]
    S = HSI[:,:,1]
    X,Y,Z = bitmap.X_Y_Z()[:,:,0],bitmap.X_Y_Z()[:,:,1],bitmap.X_Y_Z()[:,:,2]
    cv2.imshow('X',X)
    cv2.imshow('Y',Y)
    cv2.imshow('Z',Z)
    cv2.waitKey(0) """
