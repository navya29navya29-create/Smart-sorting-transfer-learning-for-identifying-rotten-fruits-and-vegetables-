import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model/model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        print("Prediction: Rotten")
    else:
        print("Prediction: Fresh")

# Example usage:
# predict_image("dataset/fresh/example.jpg")
