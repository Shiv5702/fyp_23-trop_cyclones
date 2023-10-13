import os
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from scipy.interpolate import CubicSpline
import sobel_task1
from PIL import Image
from find_coord import *
import csv
from intensity import calculate_intensity
# Directory containing DAV numpy files
dav_directory = "DAVs/"

# Path to your input CSV file
input_csv_path = 'all_clusters_with_datetime.csv'  # Replace with the actual path to your input CSV file

# Path to the output CSV file with intensity added
output_csv_path = 'your_output_csv_file.csv'  # Replace with the desired output path

# Define the function to calculate intensity
def cluster_calculate_intensity(dav_array, cluster_center_x, cluster_center_y):
    # Perform the calculations using cluster_center_x and cluster_center_y
    total = 0
    flag = False
    dav_values = []
    xy_coords = [(cluster_center_x, cluster_center_y)]  # Convert to a list of coordinates
    for x, y in xy_coords:
        total += dav_array[int(y)][int(x)]
    avg_dav = total/len(xy_coords)
    dav_values.append(avg_dav)
    calculated_intensity = []
    for dav in dav_values:
        calculated_intensity.append(calculate_intensity(dav))
    
    final_dav = dav_values[0]

    return calculated_intensity[0], flag, final_dav


# Open the input CSV file for reading
with open(input_csv_path, mode='r') as input_csvfile:
    # Create a CSV reader for the input file
    csv_reader = csv.reader(input_csvfile)

    # Read the header row
    header = next(csv_reader)

    # Add "Intensity" to the header
    header.append("Intensity")
    header.append("DAV Values")

    # Open the output CSV file for writing
    with open(output_csv_path, mode='w', newline='') as output_csvfile:
        # Create a CSV writer for the output file
        csv_writer = csv.writer(output_csvfile)

        # Write the updated header to the output file
        csv_writer.writerow(header)

        # Iterate through each row in the input CSV file
        for row in csv_reader:
            # Extract date, time, cluster_center_x, and cluster_center_y from the current row
            datetime_str = row[0] + " " + row[1] 
            cluster_center_x = float(row[3])  # Assuming it's the fourth column (zero-based index)
            cluster_center_y = float(row[4])  # Assuming it's the fifth column (zero-based index)

            # Construct the hour_str to match the DAV numpy file naming convention
            date_time = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            hour_str = date_time.strftime("%Y%m%d%H")

            # Construct the file path for the DAV numpy array
            file_path = os.path.join(dav_directory, f"merg_{hour_str}_DAV.npy")

            # Load the numpy array from the file
            dav_array = np.flipud(np.load(file_path))

            # Calculate the intensity using the provided function
            intensity = cluster_calculate_intensity(dav_array, cluster_center_x, cluster_center_y)
            if intensity[1]:
                row.append("NA")
                csv_writer.writerow(row)
            else:
                # Append the calculated intensity to the current row
                row.append(intensity[0])
                row.append(intensity[2])

                # Write the updated row to the output CSV file
                csv_writer.writerow(row)

            
