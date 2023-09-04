import os
import netCDF4
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import xarray as xr
import fsspec

# Generate image
def create_image(filename):
    # Get coordinates from netcdf4 file
    tempfile = 'resampled_file.nc4'
    dataset = load_dataset(folder + "/" + filename)
    dataset.to_netcdf(tempfile)
    nc = netCDF4.Dataset(tempfile)
    dataset.close()

    # Get the variable you want to plot
    var = nc.variables['Tb']

    # Get the latitude and longitude values
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]

    # Define the North Atlantic region (in degrees)
    lon_min, lon_max = -120, 0
    lat_min, lat_max = -5, 60

    # Find the indices of the latitude and longitude values that correspond to the desired region
    lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]

    # Find the nearest index of the minimum and maximum values
    lat_min_ind = lat_inds[0]
    lat_max_ind = lat_inds[-1]
    lon_min_ind = lon_inds[0]
    lon_max_ind = lon_inds[-1]

    # Create a 2D meshgrid of latitudes and longitudes for the desired region
    lon_subset, lat_subset = np.meshgrid(lon[lon_min_ind:lon_max_ind+1], lat[lat_min_ind:lat_max_ind+1])
    var_subset = var[0, lat_min_ind:lat_max_ind+1, lon_min_ind:lon_max_ind+1]

    # Plot the variable using Matplotlib's pcolormesh function
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.pcolormesh(lon_subset, lat_subset, var_subset, cmap='Greys')
    #im.set_clim(vmin=np.min(var_subset), vmax=280)
    ax.set_xticks(np.arange(lon_min, lon_max+15, 10))
    ax.set_yticks(np.arange(lat_min, lat_max+15, 10))
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove the axis and set the figure size to match the plot size
    ax.axis('off')
    dataname = filename[:filename.find('_', 5)]
    imgname = "Images/" + dataname + ".jpg"

    # Save the plot
    plt.savefig(imgname, bbox_inches='tight', pad_inches=0)

    # Close the NetCDF4 file
    nc.close()
    plt.close()


def load_dataset(filename, engine="h5netcdf", *args, **kwargs) -> xr.Dataset:
    """Load a NetCDF dataset from local file system or cloud bucket."""
    with fsspec.open(filename, mode="rb") as file:
        dataset = xr.load_dataset(file, engine=engine, *args, **kwargs)
    return dataset

folder = 'gs://netcdf-tropical/tropical'
dataList = open('netcdfList.txt')
allFiles = dataList.readlines()
dataList.close()

# Generate images first
for file in allFiles:
    line = file.strip()
    dataname = line[line.find('merg_'):]
    create_image(dataname)

