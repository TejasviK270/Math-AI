import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

def smooth_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        st.error("Error: Image not found.")
        return


    sigma = 1.0
    kernel = cv2.getGaussianKernel(ksize=5, sigma=sigma)
    gaussian_kernel = np.outer(kernel, kernel)

    smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(smoothed_image, cmap='gray')
    ax[1].set_title("Smoothed Image")
    ax[1].axis('off')

    st.pyplot(fig)


st.title("Gaussian Smoothing Example")
image_path = "blackwhite.jpeg"
smooth_image(image_path)

