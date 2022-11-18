import cv2
import numpy as np

CLASS_FILE_PATH = 'input/classification_classes_ILSVRC2012.txt'
CONFIG_FILE_PATH = 'models/DenseNet_121.prototxt'
MODEL_FILE_PATH = 'models/DenseNet_121.caffemodel'
IMAGE_PATH = 'input/image4.jpg' # Change image path for different images



def classifyImage(classNames,configFile,model_file,framework):
    model = cv2.dnn.readNet(model=configFile, config=model_file, framework='Caffe')
    tiger = cv2.imread(IMAGE_PATH, -1)
    blob = cv2.dnn.blobFromImage(image=tiger, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123), swapRB=False,
                                 crop=False)
    model.setInput(blob)
    outputs = model.forward()
    finalOutput = outputs[0]
    finalOutput = finalOutput.reshape(1000, 1)
    labelID = np.argmax(finalOutput)
    outputname = classNames[labelID]
    probs = np.exp(finalOutput) / np.sum(np.exp(finalOutput))
    finalProb = np.max(probs) * 100
    output = f"{outputname}, {finalProb:.3f}%"
    return output


def extractClass(filepath):
    with open(filepath, 'r') as classFile:
         classNames = classFile.read().split('\n')
    return classNames


classNames = extractClass(CLASS_FILE_PATH)
predictedClass = classifyImage(classNames, CONFIG_FILE_PATH, MODEL_FILE_PATH, 'Caffe')
print(predictedClass)




