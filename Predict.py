import tensorflow as tf
from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Step 1: Load and Prepare the MNIST Dataset for Model Training
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 2: Build the Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model
model.fit(x_train, y_train, epochs=5)

# Step 5: Evaluate the Model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Step 6: Define a function to preprocess and predict images from a directory
def predict_images_from_directory(model, directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            # Load the image
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Resize the image to 28x28 pixels
            img = cv2.resize(img, (28, 28))

            # Normalize the image
            img = img / 255.0

            # Reshape the image to match the input shape of the model
            img = np.expand_dims(img, axis=0)

            # Make a prediction
            prediction = model.predict(img)
            predicted_digit = np.argmax(prediction)

            # Display the image and prediction
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.title(f'Predicted: {predicted_digit}')
            plt.show()

            print(f'Image: {filename}, Predicted Digit: {predicted_digit}')
            print(f'Image: {filename}, Predicted Digit: {predicted_digit}')

# Step 7: Predict digits for images in the 'input_images' directory
input_directory = 'dataset\English'
# input_file = 'D:\PROJECT FINAL YEAR\another\Mine\c.jpg'
predict_images_from_directory(model, input_directory)
# predict_images_from_directory(model, input_file)
