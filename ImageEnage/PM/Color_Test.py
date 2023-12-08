
import numpy as np
import imageio

import matplotlib.pyplot as plt

def image_color_matrix(src,color_matrix):
    assert src.ndim == 3 and src.shape[-1] == 3
    assert color_matrix.ndim == 2 and color_matrix.shape[0] == 4 and color_matrix.shape[1] == 4
    srcf = np.float32(src) / 255.0
    rows,cols,chs = src.shape
    src_matrix = np.zeros((rows,cols,chs+1),dtype=np.float32)
    src_matrix[:,:,0] = srcf[:,:,0].copy()
    src_matrix[:,:,1] = srcf[:,:,1].copy()
    src_matrix[:,:,2] = srcf[:,:,2].copy()
    src_matrix[:,:,3] = 255
    dst_matrix = np.zeros_like(src_matrix)
    dst_matrix[:,:,0] = src_matrix[:,:,0] * color_matrix[0][0] + src_matrix[:,:,1] * color_matrix[0][1] + \
                        src_matrix[:,:,2] * color_matrix[0][2] + color_matrix[0][3]

    dst_matrix[:,:,1] = src_matrix[:,:,0] * color_matrix[1][0] + src_matrix[:,:,1] * color_matrix[1][1] + \
                        src_matrix[:,:,2] * color_matrix[1][2] +  color_matrix[1][3]

    dst_matrix[:,:,2] = src_matrix[:,:,0] * color_matrix[2][0] + src_matrix[:,:,1] * color_matrix[2][1] + \
                        src_matrix[:,:,2] * color_matrix[2][2] + color_matrix[2][3]

    dst_matrix[:,:,3] = src_matrix[:,:,0] * color_matrix[3][0] + src_matrix[:,:,1] * color_matrix[3][1] + \
                        src_matrix[:,:,2] * color_matrix[3][2] +  color_matrix[3][3]
    dst = np.zeros_like(srcf)
    dst[:,:,0] = dst_matrix[:,:,0]
    dst[:,:,1] = dst_matrix[:,:,1]
    dst[:,:,2] = dst_matrix[:,:,2]
    dst = np.uint8(np.clip(dst*255.0,0,255))
    return dst

src = imageio.imread('TVtest.jpg')
color_matrix = np.array([
    [1,0,0,-0.5],
    [0,-1,0,1],
    [0,0.5,-1,1],
    [0,0,0,0]
])
dst = image_color_matrix(src,color_matrix)
# print(dst)
plt.figure()
plt.imshow(src)
plt.figure()
plt.imshow(dst)

plt.show()
