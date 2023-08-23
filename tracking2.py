import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the pre-calculated DAV array
dav_array = np.load("dav_values.npy")

# Set threshold values
dav_threshold = 5.0  # Adjust this threshold as needed

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

