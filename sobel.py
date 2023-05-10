import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2

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
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction

# Load the image and convert it to a numpy array
image = cv2.imread('Screenshot 2023-05-10 at 3.57.41 pm.png')

gradient_magnitude, gradient_direction = apply_sobel_filter(image)

# Normalize the gradient magnitude for visualization
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# Convert gradient direction to degrees for visualization
gradient_direction_degrees = gradient_direction * 180 / np.pi
gradient_direction_degrees += 180

# Convert to uint8 for visualization
gradient_direction_degrees = gradient_direction_degrees.astype(np.uint8)

# Create HSV image for visualization
hsv = np.zeros_like(image)
hsv[..., 0] = gradient_direction_degrees
hsv[..., 1] = 255
hsv[..., 2] = gradient_magnitude

# Convert back to BGR for display
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


print(gradient_magnitude)
# Display the output image
cv2.imshow('Gradient', bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
