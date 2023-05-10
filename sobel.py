##The code applies the 
 # Sobel filter to calculate the gradient in the x and y directions separately, 
 # and then combines them to calculate the total gradient at each pixel. The resulting 
 # output is a numpy array with two matrices: gradient magnitude and gradient direction. 
 # The magnitude represents the strength of the gradient, and the direction represents the 
 # direction of the gradient at each pixel. The output can be visualized as arrows pointing 
 # towards the center of the storm, where larger arrows represent a higher gradient magnitude.

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
import cv2


 ##The code applies the 
 # Sobel filter to calculate the gradient in the x and y directions separately, 
 # and then combines them to calculate the total gradient at each pixel. The resulting 
 # output is a numpy array with two matrices: gradient magnitude and gradient direction. 
 # The magnitude represents the strength of the gradient, and the direction represents the 
 # direction of the gradient at each pixel. The output can be visualized as arrows pointing 
 # towards the center of the storm, where larger arrows represent a higher gradient magnitude.

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

    # Apply smoothing filter
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply Sobel filters
    gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm') / 8.0
    gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm') / 8.0

    # Calculate total gradient
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction
   
# Load the image and convert it to a numpy array
image = cv2.imread('my_plot2x.jpg')
print(apply_sobel_filter(image))

# Display the output image
# Image.fromarray(output).show()
