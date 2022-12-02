import cv2
import numpy as np
from dataPath import DATA_PATH


imagePath = DATA_PATH + "/images/Apollo-11-launch.jpg"
img = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
rotImgCV = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Rotate Image', rotImgCV)
cv2.destroyAllWindows()

imagePathMnist = DATA_PATH + "/images/MNIST_3_18x18.png"
img = cv2.imread(imagePathMnist,cv2.IMREAD_GRAYSCALE)
resizedImage = cv2.resize(img,dsize=(9,9),interpolation = cv2.INTER_AREA)
print(resizedImage)
#numpy.rot90(m, k=1, axes=(0, 1))
clockWise90rotation = np.rot90(resizedImage,axes=(1,0))
print(clockWise90rotation)
anthiclockWise90rotation = np.rot90(resizedImage)
print(anthiclockWise90rotation)
hflip = cv2.flip(resizedImage,1)
print(hflip)
