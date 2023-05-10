import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# Open the NetCDF4 file
nc = netCDF4.Dataset('DataSources/248-Data/merg_2022090506_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Create a 2D meshgrid of latitudes and longitudes
lon, lat = np.meshgrid(lon, lat)

# Plot the variable using Matplotlib's pcolormesh function
plt.figure(figsize=(10, 8))
plt.pcolormesh(lon, lat, var[0], cmap='jet')
plt.colorbar()
plt.title('Tb')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Close the NetCDF4 file
nc.close()
