import os
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import csv
from scipy.spatial.distance import euclidean

# Directory containing DAV numpy files
dav_directory = "DAVs"

# Set threshold values
dav_threshold = 2400  # Adjust this threshold as needed

# Set minimum cluster size to avoid small clusters
min_cluster_size = 50  # Adjust this value as needed

# Set the distance threshold for considering clusters as the same
distance_threshold = 10  # Adjust this threshold as needed

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

# Function to check if two cluster centers are within the distance threshold
def are_centers_close(cluster1, cluster2):
    center1_x = np.mean([point[1] for point in cluster1])
    center1_y = np.mean([point[0] for point in cluster1])
    center2_x = np.mean([point[1] for point in cluster2])
    center2_y = np.mean([point[0] for point in cluster2])
    
    distance = euclidean((center1_x, center1_y), (center2_x, center2_y))
    
    return distance <= distance_threshold

# Define the start date and time
start_datetime = datetime(2021, 8, 1, 0, 0)  # Start from August 1, 2021, at 00:00

# Define the number of hours you want to process for each day
hours_per_day = 24

# Define the number of days you want to process data for (3 months)
num_days = 92

# Create a CSV filename for the entire dataset
csv_filename = "all_clusters_with_datetime.csv"

# Write the CSV header
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Date', 'Time', 'ID', 'cluster_center_x', 'cluster_center_y'])

# Process DAV arrays and generate cluster data for specific hours and days
cluster_id = 0  # Initialize cluster ID
clusters_with_ids = []  # Store clusters with IDs
for day in range(num_days):
    for hour in range(hours_per_day):
        # Format the hour as a string with leading zeros
        hour_str = start_datetime.strftime("%Y%m%d%H")

        # Get the date and time as separate strings
        date_str = start_datetime.strftime("%Y-%m-%d")
        time_str = start_datetime.strftime("%H:%M:%S")

        # Construct the file path for the DAV numpy array
        file_path = os.path.join(dav_directory, f"merg_{hour_str}_DAV.npy")

        if not os.path.exists(file_path):
            print(f"File not found for date: {date_str} {time_str}")

        # Load the numpy array from the file
        dav_array = np.flipud(np.load(file_path))

        # Process the DAV array and generate cluster data for the specific hour
        tracked_clusters = track_clusters_bfs(dav_array, dav_threshold, min_cluster_size)
        
        # Assign IDs to clusters
        for cluster in tracked_clusters:
            assigned_id = None
            for i, existing_cluster in enumerate(clusters_with_ids):
                if are_centers_close(existing_cluster, cluster):
                    assigned_id = i
                    break
            if assigned_id is not None:
                csv_row = [date_str, time_str, assigned_id, np.mean([point[1] for point in cluster]), np.mean([point[0] for point in cluster])]
            else:
                clusters_with_ids.append(cluster)
                csv_row = [date_str, time_str, cluster_id, np.mean([point[1] for point in cluster]), np.mean([point[0] for point in cluster])]
                cluster_id += 1
            
            # Append cluster data including date and time to the CSV file
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(csv_row)

        # Increment the datetime by one hour
        start_datetime += timedelta(hours=1)
