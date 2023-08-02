
import numpy as np
import matplotlib.pyplot as plt
import cv2


def mapping_variance(varr):
    # Load the original image
    image_path = 'fyp_23-trop_cyclones\my_plot.jpg'
    original_image = cv2.imread(image_path)

    # Convert the original image to RGB format (if it's in BGR format)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Define the variations (degrees squared of deviation-variation angles)
    variations = varr

    # Create a copy of the original image to draw the variations on
    image_with_variations = np.copy(original_image)

    # Iterate over the variations and draw them on the image
    for variation in variations:
        # Generate a random color for each variation
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        
        # Draw a rectangle representing the variation on the image
        cv2.rectangle(image_with_variations, (0, 0), (variation, variation), color, -1)

    # Display the original image and the image with variations
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(image_with_variations)
    axes[1].set_title('Image with Variations')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()




