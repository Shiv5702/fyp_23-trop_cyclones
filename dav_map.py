import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_deviation_angle_variance_map(dav_values):
    # Calculate image dimensions
    image_height, image_width = dav_values.shape
    cmap = 'gray'  # Use a grayscale colormap

    # flip vertically to correlate with coordinates properly as the y-direction from image is different
    dav_values = np.flipud(dav_values)
    
    # Create a normalization instance to map DAV values
    norm = mcolors.Normalize(vmin=np.min(dav_values), vmax=np.max(dav_values))
    
    # Plotting the map of deviation-angle variances and saving the image
    plt.imshow(dav_values, cmap=cmap, norm=norm, origin='lower')
    plt.axis('off')  # Remove the axis labels and ticks
    plt.savefig("DAV_Map.jpg", bbox_inches='tight', pad_inches=0)
    plt.colorbar()
    plt.title('Map of Deviation-Angle Variances')
    plt.show()

# Generate random DAV values (dummy data)
#image_height = 100
#image_width = 100
#dav_values = np.random.rand(image_height, image_width)

# Generate and display the map
#generate_deviation_angle_variance_map(dav_values)
