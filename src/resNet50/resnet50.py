import tensorflow as tf
#IMAGE_PATH = 'input/car.jpg' # Change image path for different images
IMAGE_PATH = 'input/dog.jpg' # Change image path for different images

def preProcessing(imgPath):
    # Read image using TensorFlow.
    tfImage = tf.io.read_file(imgPath)
    # Decode the above `tf_image` from a Bytes string to a Tensor.
    decodedImage = tf.image.decode_image(tfImage)
    imageResized = tf.image.resize(decodedImage, (224, 224))
    # Add batch dimension at the beginning.
    imageBatch = tf.expand_dims(imageResized, axis=0)
    # Forward pass through the model.
    imageBatch = tf.keras.applications.imagenet_utils.preprocess_input(imageBatch)
    return imageBatch

def resNet50Predict(imgPath):
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
    imageBatch = preProcessing(IMAGE_PATH)
    predictions = model.predict(imageBatch)
    return predictions

def decodeTop5Prediction(predictions):
    # Decode the predictions from class number to actual class names.
    processedPreds = tf.keras.applications.imagenet_utils.decode_predictions(
        preds=predictions,
        top=5
    )
    return processedPreds

def showPrediction(processedPreds):
    for num, pred in enumerate(processedPreds[0]):
        print(f"Prediction {num}: {pred[1]}, {pred[2] * 100:.2f}%")



predictions = resNet50Predict(IMAGE_PATH)
processedPreds = decodeTop5Prediction(predictions)
showPrediction(processedPreds)





