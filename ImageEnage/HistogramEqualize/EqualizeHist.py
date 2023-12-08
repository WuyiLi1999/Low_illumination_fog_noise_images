#直方图均衡化（HE）图像增强
import time

import cv2
import numpy as np

from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from skimage.exposure import histogram, cumulative_distribution



def rgb_adjuster_lin(image):
    target_bins = np.arange(255)
    target_freq = np.linspace(0, 1, len(target_bins))
    freq_bins = [cumulative_distribution(image[:, :, i]) for i in \
                 range(3)]
    names = ['Reds', 'Greens', 'Blues']
    line_color = ['red', 'green', 'blue']
    adjusted_figures = []
    f_size = 20

    # Pad frequencies with min frequency
    adj_freqs = []
    for i in range(len(freq_bins)):
        if len(freq_bins[i][0]) < 256:
            frequencies = list(freq_bins[i][0])
            min_pad = [min(frequencies)] * (256 - len(frequencies))
            frequencies = min_pad + frequencies
        else:
            frequencies = freq_bins[i][0]
        adj_freqs.append(np.array(frequencies))

    # Plot RGB Images
    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    for n, ax in enumerate(ax.flatten()):
        interpolation = np.interp(adj_freqs[n], target_freq,
                                  target_bins)
        adjusted_image =img_as_ubyte(interpolation[image[:, :, n]].astype(int))
        ax.set_title(f'{names[n]}', fontsize=f_size)
        ax.imshow(adjusted_image, cmap=names[n])
        adjusted_figures.append([adjusted_image])
    fig.tight_layout()
    # Plot Adjusted CDFs
    fig, ax = plt.subplots(1, 3, figsize=[15, 3])
    for n, ax in enumerate(ax.flatten()):
        interpolation = np.interp(adj_freqs[n], target_freq,
                                  target_bins)
        adjusted_image = img_as_ubyte(interpolation[image[:, :, n]]
                                      .astype(int))
        freq_adj, bins_adj = cumulative_distribution(adjusted_image)

        ax.set_title(f'{names[n]}', fontsize=f_size)
        ax.step(bins_adj, freq_adj, c=line_color[n],
                label='Actual CDF')
        ax.plot(target_bins,
                target_freq,
                c='gray',
                label='Target CDF',
                linestyle='--')
    fig.tight_layout()

    adjusted_image = np.dstack((adjusted_figures[0][0],
                                adjusted_figures[1][0],
                                adjusted_figures[2][0]))

    # Plot Original Image against Adjusted Image
    fig, ax = plt.subplots(1, 2, figsize=[17, 7])

    ax[0].imshow(image);
    ax[0].set_title(f'Original Image', fontsize=f_size)

    ax[1].imshow(adjusted_image);
    ax[1].set_title(f'Adjusted Image', fontsize=f_size)

    fig.tight_layout()

    return adjusted_image


if __name__ == '__main__':
    begin = time.time()
    #begin=time.mktime(time.localtime())
    # #获取原图像对应的图像矩阵--RGB三通道
    # Img_Rgb = cv2.imread("waterfall.jpg")
    # Img_Grey=cv2.cvtColor(Img_Rgb,cv2.COLOR_RGB2GRAY)#RGB转化为灰度图像
    # result=myEqualizeHist(Img_Grey).astype(np.uint8)
    # Img_Rgb=cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)
    # cv2.imwrite('HistogramEqualize.jpg',Img_Rgb)
    dark_image = cv2.imread('../Image/waterfall.jpg');
    result=rgb_adjuster_lin(dark_image)
    cv2.imwrite('waterfallHE.jpg', result)
    #end=time.mktime(time.localtime())
    end = time.time()
    print(int(round(end * 1000))-int(round(begin * 1000)))