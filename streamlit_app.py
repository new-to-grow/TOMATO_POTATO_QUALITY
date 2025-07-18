import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import io

# Load model
model = load_model("vit_model.keras")

# Class names
crop_classes = ['Potato', 'Tomato']
condition_classes = ['Unhealthy', 'Healthy']

# Name of the last convolutional layer for Grad-CAM
TARGET_LAYER = 'conv2d'  # Replace with your actual conv layer name

# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, list):
            pred = predictions[0] if pred_index == 0 else predictions[1]
        else:
            pred = predictions

        top_class = tf.argmax(pred[0])
        loss = pred[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

# Overlay heatmap on original image
def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_array = np.array(img)

    # Convert heatmap to RGB if needed
    if heatmap.shape != img_array.shape:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(superimposed_img)

# Streamlit UI
st.set_page_config(page_title="Crop & Condition Classifier", layout="centered")
st.title("🌱 Crop & Condition Classifier with Grad-CAM")

# File upload
uploaded_file = st.file_uploader("Upload an image of a crop leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(pil_image)

    # Predict
    predictions = model.predict(img_array)
    crop_pred = predictions[0][0]
    condition_pred = predictions[1][0]

    crop_idx = np.argmax(crop_pred)
    condition_idx = np.argmax(condition_pred)

    st.write(f"**Crop:** {crop_classes[crop_idx]} ({crop_pred[crop_idx]*100:.2f}%)")
    st.write(f"**Condition:** {condition_classes[condition_idx]} ({condition_pred[condition_idx]*100:.2f}%)")

    # Grad-CAM heatmaps
    crop_heatmap = make_gradcam_heatmap(img_array, model, TARGET_LAYER, pred_index=0)
    condition_heatmap = make_gradcam_heatmap(img_array, model, TARGET_LAYER, pred_index=1)

    crop_overlay = overlay_heatmap(pil_image, crop_heatmap)
    condition_overlay = overlay_heatmap(pil_image, condition_heatmap)

    st.subheader("🔍 Grad-CAM Visualizations")

    # Display original + both heatmaps in one row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(pil_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(crop_overlay, caption="Crop Prediction Heatmap", use_container_width=True)

    with col3:
        st.image(condition_overlay, caption="Condition Prediction Heatmap", use_container_width=True)
