import cv2 as cv
from skimage import measure
from skimage.metrics import structural_similarity

# Calculate SSIM
first = cv.imread("imagePlus/newImage/bench.jpg")

#second = cv.imread("captured.png")
second = cv.imread("imagePlus/newImage/Ref[25].jpg")

(score,diff) = structural_similarity(first,second,multichannel=True,full=True)
diff = (diff *255).astype("uint8")
print("SSIM:{}".format(score))
# first = cv.resize(first, (2576,1125))
# second = cv.resize(second, (2576,1125))
# first = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
# second = cv.cvtColor(second, cv.COLOR_BGR2GRAY)
# s = measure.compare_ssim(first, second)
# print(s)