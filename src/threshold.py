import cv2
import numpy as np
from dataPath import DATA_PATH
from dataPath import OUTPUT_DATA_PATH
imagePath = DATA_PATH + "/images/road_lanes.png"

img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
retval,imgthresh = cv2.threshold(img,165,255,cv2.THRESH_BINARY)
cv2.imshow('Image', img)
cv2.imshow('Thresholded', imgthresh)
cv2.waitKey(2000)
cv2.destroyAllWindows()

BLOCK_SIZE = 7
CONST = 7


imagePath = DATA_PATH + "/images/Piano_Sheet_Music.png"
img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
imgThreshadap = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,BLOCK_SIZE,CONST)
cv2.imshow('Image', img)
cv2.waitKey(2000)
cv2.imshow('Thresholded', imgThreshadap)
cv2.waitKey(2000)
cv2.destroyAllWindows()


