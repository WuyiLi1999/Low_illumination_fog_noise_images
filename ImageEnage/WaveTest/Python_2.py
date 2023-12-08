import cv2
#根据指定路径获取图像
img = cv2.imread("../Image/foggy_bench.jpg")

def getImageCoordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", getImageCoordinate)
while (1):
    cv2.imshow("image", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
