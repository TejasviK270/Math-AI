import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

def smooth_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return
    
    kernel_size = (5, 5)
    sigma = 1.0
    kernel = cv2.getGaussianKernel(ksize=5, sigma=sigma)
    gaussian_kernel = np.outer(kernel, kernel)

    smoothed_image = cv2.filter2D(image, -1, gaussian_kernel)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Smoothed Image")
    plt.imshow(smoothed_image, cmap='gray')
    plt.axis('off')

    plt.show()

# Use a raw string for the image path
image_path = "blackwhite.jpeg"
smooth_image(image_path)
