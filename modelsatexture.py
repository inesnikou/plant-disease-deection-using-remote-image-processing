
import os
import numpy as np
import rasterio
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Function to load a band with resolution reduction
def load_band(i, image_folder):
    filename = os.path.join(image_folder, f"LC09_L2SP_186056_20240409_20240410_02_T1_SR_B{i}.tif")
    with rasterio.open(filename) as src:
        print(f"Loading {filename}")
        data = src.read(1)
        data_resized = cv2.resize(data, (data.shape[1] // 8, data.shape[0] // 8))  # Reduce size further
        return data_resized

# Path to the folder containing images of different bands
image_folder = r"C:\\Users\\User\\Desktop\\satellite images"
# Load data from different bands in parallel
band_data = [load_band(i, image_folder) for i in range(1, 8)]

# Convert the list to a numpy array
band_data = np.array(band_data, dtype=np.uint8)  # Use efficient data type
print("Band data dimensions:", band_data.shape)

# Extract Red (band 4) and Near-Infrared (band 5) channels
red = band_data[3]
nir = band_data[4]

# Calculate NDVI
denominator = nir + red
denominator[denominator == 0] = 1  # To avoid division by zero
ndvi = (nir - red) / denominator
print("NDVI calculated, dimensions:", ndvi.shape)

# Resize the image to reduce memory consumption
scale_factor = 0.05  # Reduce image size further to 5% of its original size
gray_image_resized = cv2.resize((red + nir) / 2, (0, 0), fx=scale_factor, fy=scale_factor)

# Initialize ImageDataGenerator
data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Paths to the dataset directories
dataset_path = r"C:\\Users\\User\\Desktop\\Project app\\tomato_dataset\\test"

# Create training and validation generators
train_gen = data_gen.flow_from_directory(dataset_path,
                                         target_size=(32, 32),
                                         color_mode='grayscale',
                                         class_mode='categorical',
                                         batch_size=32,
                                         subset='training')

val_gen = data_gen.flow_from_directory(dataset_path,
                                       target_size=(32, 32),
                                       color_mode='grayscale',
                                       class_mode='categorical',
                                       batch_size=32,
                                       subset='validation')

# Build the optimized CNN model
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),  # Reduced filters and size
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),  # Reduced dense layer size
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')  # Number of output classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with generators
history = model.fit(train_gen, epochs=115, validation_data=val_gen)

# Save the model
model.save('models/cnn_model_plant.h5')

# Evaluate the model
loss, accuracy = model.evaluate(val_gen)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict on validation data
y_val_true = val_gen.classes
y_val_pred = np.argmax(model.predict(val_gen), axis=-1)

# Print class information for debugging
print(f"y_val_true shape: {y_val_true.shape}")
print(f"y_val_pred shape: {y_val_pred.shape}")
print(f"Class labels: {list(val_gen.class_indices.keys())}")
print(f"Unique classes in y_val_true: {np.unique(y_val_true)}")
print(f"Unique classes in y_val_pred: {np.unique(y_val_pred)}")

# Confusion matrix
conf_matrix = confusion_matrix(y_val_true, y_val_pred)

# Get class labels from the validation generator
class_labels = list(val_gen.class_indices.keys())

# Ensure the class labels match the unique classes in the validation set
unique_classes = np.unique(np.concatenate([y_val_true, y_val_pred]))
class_labels = [class_labels[i] for i in unique_classes]

# Classification report
class_report = classification_report(y_val_true, y_val_pred, labels=unique_classes, target_names=class_labels, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

# Save confusion matrix and classification report
np.savetxt('results/confusion_matrix.csv', conf_matrix, delimiter=',')
class_report_df.to_csv('results/classification_report.csv')

# Save history
history_df = pd.DataFrame(history.history)
history_df.to_csv('results/training_history.csv', index=False)

print("Model training and evaluation completed.")
print("Confusion matrix and classification report saved.")

# Load the saved model for predictions
loaded_model = tf.keras.models.load_model('models/cnn_model_plant.h5')

# Function to make predictions on new images
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (32, 32))
    img_rescaled = img_resized / 255.0
    img_expanded = np.expand_dims(img_rescaled, axis=(0, -1))  # Add batch and channel dimensions
    prediction = loaded_model.predict(img_expanded)
    predicted_class = np.argmax(prediction, axis=-1)
    class_labels = list(val_gen.class_indices.keys())
    return class_labels[predicted_class[0]], prediction

# Example usage of the prediction function
test_image_path = r"C:\\Users\\User\Desktop\\Project app\\tomato_dataset\\test\\Tomato___Bacterial_spot\\0a6d40e4-75d6-4659-8bc1-22f47cdb2ca8___GCREC_Bact.Sp 6247.JPG"  # Replace with your test image path
predicted_class, prediction_probs = predict_image(test_image_path)
print(f"Predicted Class: {predicted_class}, Prediction Probabilities: {prediction_probs}")
