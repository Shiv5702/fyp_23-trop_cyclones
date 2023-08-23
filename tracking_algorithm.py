import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the pre-calculated DAV array
dav_array = np.load("fyp_23-trop_cyclones/dav_values.npy")

# Set threshold values
dav_threshold = 3000  # Adjust this threshold as needed

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

# Overlay tracked clusters on the DAV map
def overlay_clusters_on_dav_map(dav_values, clusters):
    cmap = 'jet'
    norm = mcolors.Normalize(vmin=np.min(dav_values), vmax=np.max(dav_values))
    
    plt.imshow(dav_values, cmap=cmap, norm=norm, origin='lower')
    plt.axis('off')

    # Plot the clusters as points or markers
    for cluster in clusters:
        x_coords, y_coords = zip(*cluster)
        cluster_center_x = np.mean(x_coords)
        cluster_center_y = np.mean(y_coords)
        cluster_radius = max(np.ptp(x_coords), np.ptp(y_coords)) / 2  # Compute the radius as half of the maximum extent
        circle = plt.Circle((cluster_center_y, cluster_center_x), cluster_radius, color='red', fill=False, lw=1)
        plt.gca().add_patch(circle)
        #x_coords, y_coords = zip(*cluster)
        #plt.scatter(y_coords, x_coords, color='red', s=10)  # You can adjust the color and size as needed


    plt.colorbar()
    plt.title('Tracked Clusters on DAV Map')
    plt.show()

# Overlay and visualize the tracked clusters on the DAV map
overlay_clusters_on_dav_map(dav_array, tracked_clusters)