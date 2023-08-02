import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_gradient_vectors(image, scale_factor=10):
    # Apply the Sobel filter
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Find non-zero vectors
    non_zero_indices = np.nonzero(gradient_magnitude)
    gradient_x = gradient_x[non_zero_indices]
    gradient_y = gradient_y[non_zero_indices]
    x = non_zero_indices[1]
    y = non_zero_indices[0]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image in the first subplot
    axs[0].imshow(image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Plot the gradient vectors on the image in the second subplot
    axs[1].imshow(image, cmap='gray')
    axs[1].quiver(x, y, gradient_x, gradient_y, color='red', angles='xy', scale_units='xy', scale=scale_factor)
    axs[1].axis('off')
    axs[1].set_title('Gradient Vectors')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()




# Load the image
image = cv2.imread('fyp_23-trop_cyclones\my_plot.jpg', cv2.IMREAD_GRAYSCALE)


# Plot the gradient vectors on the image with a custom scale factor
plot_gradient_vectors(image, scale_factor=150)


def find_strongest_gradient_magnitude(image):
    # Apply the Sobel filter
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Find the indices of the maximum gradient magnitude
    max_index = np.argmax(gradient_magnitude)
    max_coordinate = np.unravel_index(max_index, gradient_magnitude.shape)

    # Return the coordinates of the strongest gradient magnitude
    return max_coordinate



# Find the coordinates of the strongest gradient magnitude
coordinates = find_strongest_gradient_magnitude(image)

# Plot the original image with the marker at the strongest gradient magnitude coordinates
plt.imshow(image, cmap='gray')
plt.plot(coordinates[1], coordinates[0], 'ro', markersize=5)
plt.xlabel('Column')
plt.ylabel('Row')
plt.title('Original Image with Strongest Gradient Magnitude')
plt.show()

print("Coordinates of the Strongest Gradient Magnitude:")
print("Row:", coordinates[0])
print("Column:", coordinates[1])