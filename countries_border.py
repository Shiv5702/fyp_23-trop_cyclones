import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Open the NetCDF4 file
nc = netCDF4.Dataset('DataSources/merg_2022020200_4km-pixel.nc4')

# Get the variable you want to plot
var = nc.variables['Tb']

# Get the latitude and longitude values
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -100, 0
lat_min, lat_max = 0, 60

# Find the indices of the latitude and longitude values that correspond to the desired region
lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]

# Create a 2D meshgrid of latitudes and longitudes for the desired region
lon_subset, lat_subset = np.meshgrid(lon[lon_inds], lat[lat_inds])

# Select the variable values for the desired region
var_subset = var[0, lat_inds[0]:lat_inds[-1]+1, lon_inds[0]:lon_inds[-1]+1]

# Plot the variable using Matplotlib's pcolormesh function
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
im = ax.pcolormesh(lon_subset, lat_subset, var_subset, cmap='jet')
ax.set_xticks(np.arange(lon_min, lon_max+10, 10))
ax.set_yticks(np.arange(lat_min, lat_max+10, 10))
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
plt.colorbar(im, ax=ax, shrink=0.5)
title_date = datetime.strptime('2022020200', '%Y%m%d%H')
plt.title(f'Tb (North Atlantic) - {title_date.strftime("%Y-%m-%d %H:%M:%S")}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add country borders
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import colorConverter

# Load country borders
from urllib.request import urlopen
import json
import requests

url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
response = requests.get(url)
country_data = response.json()

country_polys = []
for feature in country_data['features']:
    if feature['geometry']['type'] == 'Polygon':
        coords = feature['geometry']['coordinates'][0]
        poly = Polygon(coords, closed=True)
        country_polys.append(poly)
    elif feature['geometry']['type'] == 'MultiPolygon':
        for subcoords in feature['geometry']['coordinates']:
            poly = Polygon(subcoords[0], closed=True)
            country_polys.append(poly)

# Define the colors for each country's border
edgecolors = [colorConverter.to_rgba('black')]*len(country_polys)

# Plot the country borders
country_patches = PatchCollection(country_polys, facecolor='none', edgecolor=edgecolors, linewidths=1, alpha=1.0)
ax.add_collection(country_patches)


plt.show()

# Close the NetCDF4 file
nc.close()
