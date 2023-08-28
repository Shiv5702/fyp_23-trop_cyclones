import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# Load the pre-calculated DAV array
dav_array = np.load("dav_values.npy")

# Load the JPEG image as a background
background_image = Image.open("my_plot.jpg")

# Set threshold values
dav_threshold = 3100  # Adjust this threshold as needed

# Define a function to perform tracking
def track_clusters(dav_array, threshold):
    height, width = dav_array.shape
    visited = np.zeros((height, width), dtype=bool)
    clusters = []

    def dfs(x, y):
        cluster = []
        stack = [(x, y)]

        while stack:
            x, y = stack.pop()
            if visited[x, y]:
                continue
            visited[x, y] = True
            cluster.append((x, y))

            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in neighbors:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < height and 0 <= new_y < width and not visited[new_x, new_y] and dav_array[new_x, new_y] >= threshold:
                    stack.append((new_x, new_y))

        return cluster

    for x in range(height):
        for y in range(width):
            if not visited[x, y] and dav_array[x, y] >= threshold:
                new_cluster = dfs(x, y)
                if len(new_cluster) > 0:
                    clusters.append(new_cluster)

    return clusters

# Perform tracking
tracked_clusters = track_clusters(dav_array, dav_threshold)

# Print the tracked clusters (you can modify this to store or visualize the results)
for idx, cluster in enumerate(tracked_clusters):
    print(f"Cluster {idx+1}: {cluster}")

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Overlay tracked clusters on the DAV map
def overlay_clusters_on_dav_map(ax, dav_values, clusters):
    cmap = 'jet'
    norm = mcolors.Normalize(vmin=np.min(dav_values), vmax=np.max(dav_values))

    ax.imshow(dav_values, cmap=cmap, norm=norm, origin='upper')
    ax.axis('off')
    ax.set_title('DAV Map')

    # Plot the clusters as points or markers with circles
    for cluster in clusters:
        x_coords, y_coords = zip(*cluster)
        cluster_center_x = np.mean(x_coords)
        cluster_center_y = np.mean(y_coords)
        cluster_radius = max(np.ptp(x_coords), np.ptp(y_coords)) / 2
        circle = plt.Circle((cluster_center_y, cluster_center_x), cluster_radius, color='red', fill=False, lw=1)
        ax.add_patch(circle)

# Overlay tracked clusters on the flipped JPEG image
def overlay_clusters_on_image(ax, background_image, clusters):
    img_array = np.array(background_image)
    ax.imshow(img_array, origin='upper')  # Ensure origin is 'upper' for consistency
    ax.axis('off')
    ax.set_title('Clusters on Image')

    # Plot the clusters as points or markers with circles
    for cluster in clusters:
        x_coords, y_coords = zip(*cluster)
        cluster_center_x = np.mean(x_coords)
        cluster_center_y = np.mean(y_coords)
        cluster_radius = max(np.ptp(x_coords), np.ptp(y_coords)) / 2
        circle = plt.Circle((cluster_center_y, cluster_center_x), cluster_radius, color='red', fill=False, lw=1)
        ax.add_patch(circle)

# Overlay and visualize the tracked clusters on both the DAV map and the flipped JPEG image
overlay_clusters_on_dav_map(axs[0], dav_array, tracked_clusters)
overlay_clusters_on_image(axs[1], background_image, tracked_clusters)

plt.tight_layout()
plt.show()


