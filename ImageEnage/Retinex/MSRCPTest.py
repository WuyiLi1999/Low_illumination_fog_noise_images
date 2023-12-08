
#引入了均值和均方差的概念
import numpy as np
import cv2


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def simple_color_balance(input_img, s1, s2):
    h, w = input_img.shape[:2]
    temp_img = input_img.copy()
    one_dim_array = temp_img.flatten()
    sort_array = sorted(one_dim_array)

    per1 = int((h * w) * s1 / 100)
    minvalue = sort_array[per1]

    per2 = int((h * w) * s2 / 100)
    maxvalue = sort_array[(h * w) - 1 - per2]

    # 实施简单白平衡算法
    if (maxvalue <= minvalue):
        out_img = np.full(input_img.shape, maxvalue)
    else:
        scale = 255.0 / (maxvalue - minvalue)
        out_img = np.where(temp_img < minvalue, 0,temp_img)   # 防止像素溢出
        out_img = np.where(out_img > maxvalue, 255,out_img)   # 防止像素溢出
        out_img = scale * (out_img - minvalue)                # 映射中间段的图像像素
        out_img = cv2.convertScaleAbs(out_img)
    return out_img


def MSRCP(img, scales, s1, s2):
    h, w = img.shape[:2]
    scales_size = len(scales)
    B_chan = img[:, :, 0]
    G_chan = img[:, :, 1]
    R_chan = img[:, :, 2]
    log_R = np.zeros((h, w), dtype=np.float32)
    array_255 = np.full((h, w),255.0,dtype=np.float32)

    I_array = (B_chan + G_chan + R_chan) / 3.0
    I_array = replaceZeroes(I_array)

    for i in range(0, scales_size):
        L_blur = cv2.GaussianBlur(I_array, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_I = cv2.log(I_array/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_ixl = cv2.multiply(dst_I, dst_Lblur)
        log_R += cv2.subtract(dst_I, dst_ixl)
    MSR = log_R / 3.0
    Int1 = simple_color_balance(MSR, s1, s2)

    B_array = np.maximum(B_chan,G_chan,R_chan)
    A = np.minimum(array_255 / B_array, Int1/I_array)
    R_channel_out = A * R_chan
    G_channel_out = A * G_chan
    B_channel_out = A * B_chan

    MSRCP_Out_img = cv2.merge([B_channel_out, G_channel_out, R_channel_out])
    MSRCP_Out = cv2.convertScaleAbs(MSRCP_Out_img)

    return MSRCP_Out

if __name__ == '__main__':
    img = '../Image/foggy_bench.jpg'
    scales = [15,101,301]
    s1, s2 = 2,3
    src_img = cv2.imread(img)
    result = MSRCP(src_img, scales, s1, s2)

    cv2.imwrite('MSRCPbench.jpg',result)
