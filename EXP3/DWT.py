import numpy as np
def haar_wt(src, depth=3):
    [H, W] = src.shape

    count = 1
    img_tmp = np.array(src).astype(np.float32)
    wavelet = np.ones(src.shape).astype(np.float32)
    while count <= depth:
        tmpH = H // count
        tmpW = W // count
        tmp = np.zeros(src.shape).astype(np.float32)
        for i in range(0, tmpH):
            for j in range(0, tmpW // 2):
                tmp[i, j] = (img_tmp[i, 2 * j] + img_tmp[i, 2 * j + 1]) / 2.
                tmp[i, j + tmpW // 2] = (img_tmp[i, 2 * j] - img_tmp[i, 2 * j + 1]) / 2.
        
        for i in range(0, tmpH // 2):
            for j in range(0, tmpW):
                wavelet[i, j] = (tmp[2 * i, j] + tmp[2 * i + 1, j]) / 2.
                wavelet[i + tmpH // 2, j] = (tmp[2 * i, j] - tmp[2 * i + 1, j]) / 2.

        count += 1
        img_tmp = np.array(wavelet).astype(np.float32)

    return wavelet

def haar_iwt(wt, depth=3):
    [H, W] = wt.shape

    count = depth
    img_tmp = np.array(wt).astype(np.float32)
    src = np.zeros(wt.shape).astype(np.float32)
    while count >= 1:
        tmpH = H // count
        tmpW = W // count
        tmp = np.zeros(wt.shape).astype(np.float32)

        for i in range(0, tmpH // 2):
            for j in range(0, tmpW):
                tmp[2 * i, j] = img_tmp[i, j] + img_tmp[i + tmpH // 2, j]
                tmp[2 * i + 1, j] = img_tmp[i, j] - img_tmp[i + tmpH // 2, j]

        src = np.array(img_tmp).astype(np.float32)

        for i in range(0, tmpH):
            for j in range(0, tmpW // 2):
                src[i, 2 * j] = tmp[i, j] + tmp[i, j + tmpW // 2]
                src[i, 2 * j + 1] = tmp[i, j] - tmp[i, j + tmpW // 2]

        count -= 1
        img_tmp = np.array(src).astype(np.float32)

    return src
def denosie_DWT(wt, noise_sigma=11.0, depth=3):

    wtt = wt.copy()
    UP_hold = noise_sigma * np.sqrt(2 * np.log2(wtt.size))
    for i in range(0, wtt.shape[0]):
        for j in range(0, wtt.shape[1]):
            if np.abs(wtt[i, j]) < UP_hold:
                wtt[i, j] = 0.0

    return haar_iwt(wtt, depth=depth)