import numpy as np
import csv

# Load the .npy file
npy_file_path = 'dav_values.npy'
data = np.load(npy_file_path)

# Flatten the array if needed
flattened_data = data.flatten()

# Define the CSV output file path
csv_file_path = 'dav_values.csv'

# Write the data to a CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write each value as a row in the CSV file
    for value in flattened_data:
        csv_writer.writerow([value])
