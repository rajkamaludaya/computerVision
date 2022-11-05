import cv2
import numpy as np
from dataPath import DATA_PATH
from dataPath import OUTPUT_DATA_PATH

imagePath = DATA_PATH + "/images/Apollo-8-launch.png"
img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Display Image", img)
c = cv2.waitKey(2000)
cv2.destroyAllWindows()

outputPath = OUTPUT_DATA_PATH + "/images/Apollo-11-launch.png"
cv2.imwrite(outputPath, img)
