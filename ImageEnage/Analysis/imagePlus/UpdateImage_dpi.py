import os

import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    path = "PlosImage/" #设置文件路径
    images = os.listdir(path)  #遍历指定文件夹
    for num in range(300, 601, 100):

        for image_name in images:
            image_path = os.path.join(path, image_name) #构建文件夹中的每一个文件路径
            plt.imshow(cv2.imread(image_path))
            plt.axis('off')  # 保存图片中不带边框

            plt.xticks([])  # 保存的图片不显示 x，y 轴的刻度
            plt.yticks([])

            index = image_name.index(".") #截取子串（获取文件名）
            newPath = 'C:/Users/Smile/Desktop/Plos/dpi' + str(num) + '/'+ image_name[:index+1]+'tiff' #构建文件报存的路径
            print(newPath)
            plt.savefig(newPath, #文件保存的路径
                        dpi=300,  #dpi值
                        format="tiff",bbox_inches='tight') #图片保存的模式是紧凑的，即图片两边没有多余的空白

