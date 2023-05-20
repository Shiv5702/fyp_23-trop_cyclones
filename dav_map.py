import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_deviation_angle_variance_map(dav_values):
    # Calculate image dimensions
    image_height, image_width = dav_values.shape
    cmap = 'gray'  # Use a grayscale colormap
    
    # Create a normalization instance to map DAV values
    norm = mcolors.Normalize(vmin=np.min(dav_values), vmax=np.max(dav_values))
    
    # Plotting the map of deviation-angle variances
    plt.imshow(dav_values, cmap=cmap, norm=norm, origin='lower')
    plt.colorbar()
    plt.title('Map of Deviation-Angle Variances')
    plt.axis('off')  # Remove the axis labels and ticks
    plt.show()

# Generate random DAV values (dummy data)
#image_height = 100
#image_width = 100
#dav_values = np.random.rand(image_height, image_width)

# Generate and display the map
#generate_deviation_angle_variance_map(dav_values)
