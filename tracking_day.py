import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime, timedelta
from collections import deque
from scipy import ndimage

# Directory containing DAV numpy files
dav_directory = "fyp_23-trop_cyclones/DAVs"
image_directory = "fyp_23-trop_cyclones/Images"

# Set threshold values
dav_threshold = 2400  # Adjust this threshold as needed

# Set minimum cluster size to avoid small circles
min_cluster_size = 50  # Adjust this value as needed

# Function to perform tracking
def track_clusters_bfs(dav_array, fixed_threshold, min_cluster_size):
    
    height, width = dav_array.shape
    visited = np.zeros((height, width), dtype=bool)
    clusters = []

    def bfs(x, y):
        cluster = []
        queue = deque([(x, y)])

        while queue:
            x, y = queue.popleft()
            if visited[x, y]:
                continue
            visited[x, y] = True
            cluster.append((x, y))

            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in neighbors:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < height and 0 <= new_y < width and not visited[new_x, new_y] and dav_array[new_x, new_y] <= fixed_threshold:
                    queue.append((new_x, new_y))

        return cluster

    for x in range(height):
        for y in range(width):
            if not visited[x, y] and dav_array[x, y] <= fixed_threshold:
                new_cluster = bfs(x, y)
                if len(new_cluster) >= min_cluster_size:
                    clusters.append(new_cluster)

    return clusters


def track_clusters_dfs(dav_array, threshold):
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
                if 0 <= new_x < height and 0 <= new_y < width and not visited[new_x, new_y] and dav_array[new_x, new_y] <= threshold:
                    stack.append((new_x, new_y))

        return cluster

    for x in range(height):
        for y in range(width):
            if not visited[x, y] and dav_array[x, y] <= threshold:  # Change the threshold condition to find below threshold
                new_cluster = dfs(x, y)
                if len(new_cluster) > 0:
                    clusters.append(new_cluster)

    return clusters

# Function to process and plot a single DAV image
def process_and_plot_single_dav_array(dav_array, hour, image_path):
    # Load the corresponding image
    image = cv2.imread(image_path)

    # Create subplots for Clusters and the corresponding Image
    plt.figure(figsize=(8, 6))

    # Define the custom colormap and normalization
    # cmap = 'jet'
    #cmap = plt.get_cmap('jet')
    cmap = plt.get_cmap('jet')  # Use 'jet_r' to reverse the 'jet' colormap
    
    norm = plt.Normalize(vmin=np.min(dav_array), vmax=np.max(dav_array))
    #norm = plt.Normalize(vmin=2100, vmax=3200)


    # Subplot 1: Clusters
    plt.subplot(1, 2, 1)
    plt.imshow(dav_array, cmap=cmap, norm=norm, origin='upper')
    tracked_clusters = track_clusters_bfs(dav_array, dav_threshold, min_cluster_size)
    for cluster in tracked_clusters:
        if len(cluster) >= min_cluster_size:
            x_coords, y_coords = zip(*cluster)
            cluster_center_x = np.mean(x_coords)
            cluster_center_y = np.mean(y_coords)
            cluster_radius = max(np.ptp(x_coords), np.ptp(y_coords)) / 2
            circle = plt.Circle((cluster_center_y, cluster_center_x), cluster_radius, color='red', fill=False, lw=1)
            plt.gca().add_patch(circle)
    plt.title("Clusters")

    # Subplot 2: Image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the loaded image
    plt.title("Image")

    # Remove axis labels
    for ax in plt.gcf().get_axes():
        ax.axis('off')

    # Set the entire date as the figure title
    plt.suptitle(datetime_str)

    # Adjust subplot layout
    plt.tight_layout()

    # Display the plot
    plt.show()

# Define the start date and time
start_datetime = datetime(2021, 8, 11, 0, 0)

# Define the number of hours you want to process
num_hours = 24

# Process and plot DAV arrays and images for specific hours
for hour in range(num_hours):
    # Format the hour as a string with leading zeros
    hour_str = start_datetime.strftime("%Y%m%d%H")

    # Get the entire date as a string
    datetime_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct the file path for the DAV numpy array
    file_path = os.path.join(dav_directory, f"merg_{hour_str}_DAV.npy")
    
    # Load the numpy array from the file
    dav_array = np.load(file_path)
    
    # Construct the file path for the corresponding image
    image_path = os.path.join(image_directory, f"merg_{hour_str}.jpg")
    
    # Process and plot the DAV array and image for the specific hour
    process_and_plot_single_dav_array(dav_array, hour, image_path)
    
    # Increment the datetime by one hour
    start_datetime += timedelta(hours=1)
