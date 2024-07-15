import streamlit as st
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from skimage import exposure, segmentation
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import tempfile

# Function to save an image using rasterio
def save_image(data, filepath):
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        filepath, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype, transform=transform
    ) as dst:
        dst.write(data, 1)

# Function to load a band with resolution reduction
def load_band(uploaded_file, save_folder, band_num):
    file_path = uploaded_file.name
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    with rasterio.open(temp_file_path) as src:
        st.write(f"Loading {file_path}")
        data = src.read(1)
        original_save_path = os.path.join(save_folder, f"band_{band_num}_original.tif")
        save_image(data, original_save_path)
        data_resized = cv2.resize(data, (data.shape[1] // 8, data.shape[0] // 8))
        resized_save_path = os.path.join(save_folder, f"band_{band_num}_resized.tif")
        save_image(data_resized, resized_save_path)
        return data_resized

# Function to convert TIFF images to PNG
def convert_tiff_to_png(tiff_path, png_path):
    with rasterio.open(tiff_path) as src:
        data = src.read(1)
        data_uint8 = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)
        img = Image.fromarray(data_uint8)
        img.save(png_path)

# Load the model
try:
    model = tf.keras.models.load_model(r'C:\\Users\\User\\Desktop\\Project app\\models\\cnn_model_plant.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")

st.set_page_config(layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Preprocessing", "Evaluation Metrics", "Predictions"]
page = st.sidebar.radio("Go to", pages)

# Predefined class labels for soil texture
class_labels = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_spot', 'Tomato_Mosaic_virus', 'Tomato_Yellow_leaf_curl_virus']

# Initialize session state variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
if 'band_data' not in st.session_state:
    st.session_state.band_data = None

if page == "Home":
    st.title("Tomato Deseasis Detection Application")
    uploaded_files = st.sidebar.file_uploader("Upload band images", accept_multiple_files=True, type=["tif"])
    if uploaded_files and st.sidebar.button("Start Preprocessing"):
        st.session_state.uploaded_files = uploaded_files
        st.session_state.preprocessed = False
        st.experimental_rerun()

elif page == "Preprocessing":
    if not st.session_state.get('uploaded_files'):
        st.warning("Please upload the band images on the Home page.")
    else:
        uploaded_files = st.session_state.uploaded_files
        save_folder = "processed_images"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        st.title("Preprocessing Page")
        if st.button("Start Preprocessing"):
            band_data = [load_band(uploaded_files[i], save_folder, i + 1) for i in range(len(uploaded_files))]
            
            if any(band is None for band in band_data):
                st.error("Error loading one or more bands.")
            else:
                band_data = np.array(band_data, dtype=np.uint8)
                st.session_state.band_data = band_data
                
                red = band_data[3]
                nir = band_data[4]
                denominator = nir + red
                denominator[denominator == 0] = 1
                ndvi = (nir - red) / denominator
                
                ndvi_save_path = os.path.join(save_folder, "NDVI.tif")
                save_image(ndvi, ndvi_save_path)
                
                scale_factor = 0.05
                gray_image_resized = cv2.resize((red + nir) / 2, (0, 0), fx=scale_factor, fy=scale_factor)
                gray_image_resized = exposure.rescale_intensity(gray_image_resized)
                
                preprocessed_save_path = os.path.join(save_folder, "preprocessed_image.tif")
                save_image(gray_image_resized, preprocessed_save_path)
                
                regions = segmentation.felzenszwalb(gray_image_resized, scale=100, sigma=0.5, min_size=50)
                segmented_save_path = os.path.join(save_folder, "segmented_image.tif")
                save_image(regions, segmented_save_path)
                
                ndvi_png_path = os.path.join(save_folder, "NDVI.png")
                convert_tiff_to_png(ndvi_save_path, ndvi_png_path)
                
                preprocessed_png_path = os.path.join(save_folder, "preprocessed_image.png")
                convert_tiff_to_png(preprocessed_save_path, preprocessed_png_path)
                
                segmented_png_path = os.path.join(save_folder, "segmented_image.png")
                convert_tiff_to_png(segmented_save_path, segmented_png_path)
                
                st.session_state.ndvi_png_path = ndvi_png_path
                st.session_state.preprocessed_png_path = preprocessed_png_path
                st.session_state.segmented_png_path = segmented_png_path
                st.session_state.preprocessed = True
                
                st.experimental_rerun()

        def display_images(image_paths, captions, width=150):
            images = [Image.open(path).convert("RGB") for path in image_paths]
            st.image(images, caption=captions, width=width)

        if st.session_state.get('preprocessed'):
            display_images([st.session_state.ndvi_png_path], ["NDVI Image"])
            display_images([st.session_state.preprocessed_png_path], ["Preprocessed Image"])
            display_images([st.session_state.segmented_png_path], ["Segmented Image"])

elif page == "Evaluation Metrics":
    if not st.session_state.get('preprocessed'):
        st.warning("Please complete preprocessing first.")
    else:
        st.title("Evaluation Metrics Page")
        if st.button("Show Evaluation Metrics"):
            history_df = pd.read_csv('results/training_history.csv')
            fig, ax = plt.subplots()
            ax.plot(history_df['loss'], label='Loss')
            ax.plot(history_df['val_loss'], label='Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
            
            class_report_df = pd.read_csv('results/classification_report.csv', index_col=0)
            st.write(class_report_df)

            conf_matrix = np.loadtxt('results/confusion_matrix.csv', delimiter=',')
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

            accuracy = history_df['val_accuracy'].iloc[-1]
            st.write(f"Model Accuracy: {accuracy:.2f}")

elif page == "Predictions":
    if not st.session_state.get('preprocessed'):
        st.warning("Please complete preprocessing first.")
    else:
        st.title("Predictions Page")
        if st.button("Generate Predictions"):
            resized_image = cv2.resize(st.session_state.band_data[3], (32, 32)).reshape(-1, 32, 32, 1)
            # predictions = model.predict(resized_image)
            texture_predictions = model.predict(resized_image)
            texture_pred_class = np.argmax(texture_predictions, axis=-1)
            
            st.subheader("Predictions")
            columns = ['Tomato desease prediction']
            texture_label = class_labels[texture_pred_class[0]]
            predictions = texture_label
            pred_df = pd.DataFrame([predictions], columns=columns)
            st.write(pred_df)

st.sidebar.write("Developed by [TATCHOU MARTINI]")
