import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('./Set_2/calib_2cam_01_2/Test/left_eye_nice.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('./Set_2/calib_2cam_03_2/Test/right_eye_nice.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('orb_nice_match.jpg', img3)
plt.grid(True)
plt.imshow(img3),plt.show()
