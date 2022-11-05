import cv2
import numpy as np
from dataPath import DATA_PATH

imagePath = DATA_PATH + "/images/img_bw_18x18.png"
bw_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Display Image", bw_img)
c = cv2.waitKey(5000)
cv2.destroyAllWindows()


imagePath = DATA_PATH + "/images/img_bw_64x64.png"
bw_img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Display Image", bw_img)
c = cv2.waitKey(5000)
cv2.destroyAllWindows()



