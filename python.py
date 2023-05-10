import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def apply_sobel_filter(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Convert to grayscale if needed
    if image.ndim == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Apply Sobel filters
    gradient_x = convolve2d(image, sobel_x, mode='same', boundary='symm') / 8.0
    gradient_y = convolve2d(image, sobel_y, mode='same', boundary='symm') / 8.0

    # Calculate total gradient
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    np.savetxt("gradient.csv", gradient_magnitude, delimiter=",")

    return gradient_magnitude

# Load the image and convert it to a numpy array
input_image = np.array(Image.open("my_plot2x.jpg"))

# Apply Sobel filter
output = apply_sobel_filter(input_image)

print(output)

# Display the output image
Image.fromarray(output).show()