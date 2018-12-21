import numpy as np
xSobel_kernel = np.array([(-1,0,1),(-2,0,2),(-1,0,1)])
ySobel_kernel = np.array([(-1,-2,-1),(0,0,0),(1,2,1)])
def get_XY_Sobel(src):
    kernel_size = ySobel_kernel.shape[0]
    src_cpy = src.copy()
    src_y = np.zeros_like(src_cpy)
    src_x = np.zeros_like(src_cpy)
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
                
            src_x[i][j] = np.sum(np.sum(img[ri:ri+kernel_size,rj:rj+kernel_size]*xSobel_kernel,axis=0),axis=0)
            src_y[i][j] = np.sum(np.sum(img[ri:ri+kernel_size,rj:rj+kernel_size]*ySobel_kernel,axis=0),axis=0)
            # print(j)
            src_cpy[i][j] = np.sqrt(src_x[i][j]**2+src_x[i][j]**2)
            j += 1
            # print(rj)
        i += 1
    return src_x,src_y,src_cpy
