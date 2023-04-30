import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

# Set up the file path and variable name
file_path = "DataSources\merg_2022020200_4km-pixel.nc4"

# Open the netcdf file
dataset = nc.Dataset(file_path)

# Print the available variable names
print(dataset.variables.keys())

# Get the longitude and latitude variables
lon = dataset.variables["lon"]
lat = dataset.variables["lat"]

# Create a new figure
fig = plt.figure()

# Plot the latitude and longitude variables
plt.plot(lon, lat)

# Set the x and y limits
plt.xlim(-180, 180)
plt.ylim(-90, 90)

# Add x and y labels
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Show the plot
plt.show()
