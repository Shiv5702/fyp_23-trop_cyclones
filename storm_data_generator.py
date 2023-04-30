import netCDF4
import numpy as np

# Open the NetCDF4 file
nc = netCDF4.Dataset('fyp_23-trop_cyclones/DataSources/merg_2022020200_4km-pixel.nc4')
# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Reshape the Tb array to match the lat/lon shape
var_2d = np.reshape(var[0], (lat.shape[0], lon.shape[0]))

# Reshape the lat and lon arrays to match the Tb shape
lat_2d = np.tile(lat, (lon.shape[0], 1)).T
lon_2d = np.tile(lon, (lat.shape[0], 1))

# Save the Tb, lat, and lon values to a CSV file
np.savetxt('tb_lat_lon.csv', np.column_stack((var_2d.flatten(), lat_2d.flatten(), lon_2d.flatten())), delimiter=',')

# Close the NetCDF4 file
nc.close()