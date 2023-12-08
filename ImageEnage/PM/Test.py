import time

import cv2
import numpy as np
# import SpiderTest.ImageEnage.PM.PMDiffusion
# import SpiderTest.ImageEnage.PM.TotalVariation
# import SpiderTest.ImageEnage.PM.HeatEquation



# img = 'waterfall.jpg' ../ACETest/Forest7ACE.jpg
from ImageEnage.PM import PMDiffusion, HeatEquation, TotalVariation
#zmIce.jpg
img = "../Image/tiananmen.png"#../ACETest/ACEDoor.jpg
src_img = cv2.imread(img)

array_img = np.array(src_img)

HE_result_array = HeatEquation._he(array_img).astype(np.float32)
begin = time.time()
Pm_result_array = PMDiffusion._pm(array_img).astype(np.float32)
end = time.time()
print(int(round(end * 1000)) - int(round(begin * 1000)))
begin = time.time()
TV_result_array = TotalVariation._tv(array_img).astype(np.float32)
end = time.time()
print(int(round(end * 1000)) - int(round(begin * 1000)))
# cv2.imwrite("HEtest_DoorACE.jpg", HE_result_array)
# cv2.imwrite("PMtest_DoorACE.jpg", Pm_result_array)
# cv2.imwrite("TVtest_DoorACE.jpg", TV_result_array)
cv2.imwrite("HEtest_Door.jpg", HE_result_array)
cv2.imwrite("PMtest_Door.jpg", Pm_result_array)
cv2.imwrite("TVtest_Door.jpg", TV_result_array)
