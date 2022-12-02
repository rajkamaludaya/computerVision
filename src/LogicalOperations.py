import cv2
import numpy as np

baseImage = np.uint8(np.ones(100,dtype=np.uint8).reshape(10,10)) * 127
print(baseImage.shape)
print(baseImage)

image2 = np.uint8(np.ones(100,dtype=np.uint8).reshape(10,10)) *255
print(image2)

finalImage = cv2.bitwise_and(baseImage,image2,mask=None)
print(finalImage)

finalImage = cv2.bitwise_or(baseImage,image2,mask=None)
print(finalImage)

finalIamge = cv2.bitwise_xor(baseImage,image2,mask=None)
print(finalIamge)



