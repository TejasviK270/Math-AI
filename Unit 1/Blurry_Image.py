import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

# Function to smooth an image
def smooth_image(image):
    # Create Gaussian kernel manually
    sigma = 1.0
    kernel = cv2.getGaussianKernel(ksize=5, sigma=sigma)
    gaussian_kernel = np.outer(kernel, kernel)

    smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)
    return smoothed_image

# Streamlit UI
st.title("Gaussian Image Smoothing")
st.write("Upload an image to apply Gaussian blur.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image in grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    smoothed = smooth_image(image)

    # Display images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(smoothed, cmap='gray')
    ax[1].set_title("Smoothed Image")
    ax[1].axis('off')

    st.pyplot(fig)
else:
    st.info("Please upload an image file.")
