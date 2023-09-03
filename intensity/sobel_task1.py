import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2
from scipy import ndimage

def apply_sobel_filter(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Convert to grayscale if needed
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian smoothing filter for denoising
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply Sobel filters
    gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm') / 8.0
    gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm') / 8.0

    # Calculate total gradient magnitude and direction
    #gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    #gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_x, gradient_y

# Load the image and convert it to a numpy array
image = cv2.imread('my_plot.jpg')

gradient_magnitude, gradient_direction = apply_sobel_filter(image)

def sobel_with_cv2(image):
    # Convert the image to grayscale if necessary
     if image.layers > 2:
         #image = np.mean(image, axis=2)
         image = np.dot(np.array(image)[...,:3], [0.2989, 0.5870, 0.1140])


     # Compute the gradient in the x and y directions
     gradient_x = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)
     gradient_y = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)

     return gradient_x, gradient_y

def another_sobel(image):

     # Convert the image to grayscale if necessary
     if image.layers > 2:
         image = np.mean(image, axis=2)

     # Compute the gradient in the x and y directions
     gradient_x = ndimage.sobel(image, axis=1)
     gradient_y = ndimage.sobel(image, axis=0)

     # Compute the gradient magnitude
     gradient_magnitude = np.hypot(gradient_x, gradient_y)

     # Compute the gradient vectors
     gradient_vectors = np.arctan2(gradient_y, gradient_x)

     return gradient_x, gradient_y


def calculate_brightness_gradient(image):
    # Convert the image to grayscale if needed
    if image.layers > 2:
        image = np.mean(image, axis=2)

    # Calculate the gradient using scipy's gradient function
    gradient_x, gradient_y = np.gradient(image)

    # Calculate the magnitude and direction of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    #return gradient_magnitude, gradient_direction
    return gradient_x, gradient_y



