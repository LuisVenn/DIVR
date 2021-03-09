import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('ESQ.jpg',cv.IMREAD_COLOR)
cv.imshow("PreView",img)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation

img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imwrite('C:/Users/Luis/Desktop/Venn/Feup/2020-21/PDI/testes/ESQ_ORB.jpg', img2)
plt.imshow(img2), plt.show()