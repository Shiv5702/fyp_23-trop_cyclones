import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime, timedelta

# Directory containing DAV numpy files
dav_directory = "DAVs/"

# Set threshold values
dav_threshold = 2850  # Adjust this threshold as needed

# Set minimum cluster size to avoid small circles
min_cluster_size = 50  # Adjust this value as needed

# Function to perform tracking
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

# Function to process and plot a single DAV image
def process_and_plot_single_dav_array(dav_array, hour):
    # Create a figure
    plt.figure(figsize=(8, 6))

    # Define the custom colormap and normalization
    cmap = 'jet'
    norm = plt.Normalize(vmin=np.min(dav_array), vmax=np.max(dav_array))

    # Display the grayscale image with the custom colormap and normalization
    plt.imshow(dav_array, cmap=cmap, norm=norm, origin='upper')

    # Plot the clusters as red circles, avoiding small clusters
    tracked_clusters = track_clusters(dav_array, dav_threshold)
    for cluster in tracked_clusters:
        if len(cluster) >= min_cluster_size:
            x_coords, y_coords = zip(*cluster)
            cluster_center_x = np.mean(x_coords)
            cluster_center_y = np.mean(y_coords)
            cluster_radius = max(np.ptp(x_coords), np.ptp(y_coords)) / 2
            circle = plt.Circle((cluster_center_y, cluster_center_x), cluster_radius, color='red', fill=False, lw=1)
            plt.gca().add_patch(circle)

    # Add a color scale (colorbar) to the plot
    cbar = plt.colorbar(orientation='vertical')
    cbar.set_label('DAV Values')

    # Remove axis labels
    plt.axis('off')
    plt.tight_layout()
    # Set the entire date as the title
    plt.title(datetime_str)
    # Save the plot as an image file without specifying a directory
    #plt.savefig(f"{datetime_str}.png")
    plt.show()

# Define the start date and time
start_datetime = datetime(2021, 8, 11, 0, 0)

# Define the number of hours you want to process
num_hours = 24

# Process and plot each DAV array one by one
for hour in range(num_hours):
    # Format the hour as a string with leading zeros
    hour_str = start_datetime.strftime("%Y%m%d%H")

    # Get the entire date as a string
    datetime_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    
    # Construct the file path
    file_path = os.path.join(dav_directory, f"merg_{hour_str}_DAV.npy")
    
    # Load the numpy array from the file
    dav_array = np.load(file_path)
    
    # Process and plot the single DAV array
    process_and_plot_single_dav_array(dav_array, hour)
    
    # Increment the datetime by one hour
    start_datetime += timedelta(hours=1)
