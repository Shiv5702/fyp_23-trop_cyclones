
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def mapping_variance(filename):
    cmap = 'jet'

    # flip vertically to correlate with coordinates properly as the y-direction from image is different
    dav_values = np.flipud(np.load("DAVs/" + filename))
    
    # Create a normalization instance to map DAV values
    norm = mcolors.Normalize(vmin=np.min(dav_values), vmax=np.max(dav_values))
    
    # Plotting the map of deviation-angle variances and saving the image
    plt.imshow(dav_values, cmap=cmap, norm=norm, origin='lower')
    plt.axis('off')  # Remove the axis labels and ticks
    plt.colorbar()
    plt.savefig("ImagesDAV/" + filename[:filename.find('.')] + ".jpg")
    plt.close()


files = os.listdir("DAVs")
for filename in files:
    mapping_variance(filename)