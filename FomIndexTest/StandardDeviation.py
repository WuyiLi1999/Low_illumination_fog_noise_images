#计算图像的均值和标准差

import cv2, os, argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    return args

def main(fileName):
    m_list, s_list = [], []
    img = cv2.imread(fileName)
    img = img / 255.0
    m, s = cv2.meanStdDev(img)
    m_list.append(m.reshape((3,)))
    s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])

if __name__ == '__main__':
    main('patents/Forest7.jpg')


