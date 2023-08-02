import netCDF4
import numpy as np

def resample_nc4_file(input_file, output_file, scale_factor):
    # Open the input NetCDF4 file
    with netCDF4.Dataset(input_file) as nc_in:
        # Read the brightness temperature variable
        brightness_temp = nc_in.variables['Tb'][:]

        # Get the latitude and longitude values
        lat = nc_in.variables['lat'][:]
        lon = nc_in.variables['lon'][:]

        # Calculate the original resolution
        original_resolution = np.abs(lat[1] - lat[0])

        # Calculate the new resolution
        new_resolution = original_resolution * scale_factor

        # Calculate the indices to select the data for resampling
        indices = np.arange(0, brightness_temp.shape[1], scale_factor)

        # Resample the brightness temperature data
        resampled_brightness_temp = brightness_temp[:, indices, :][:, :, indices]

        # Open the output NetCDF4 file for writing
        with netCDF4.Dataset(output_file, 'w') as nc_out:
            # Create dimensions and variables in the output file
            nc_out.createDimension('time', None)
            nc_out.createDimension('lat', resampled_brightness_temp.shape[1])
            nc_out.createDimension('lon', resampled_brightness_temp.shape[2])

            time_var = nc_out.createVariable('time', 'i4', ('time',))
            lat_var = nc_out.createVariable('lat', 'f4', ('lat',))
            lon_var = nc_out.createVariable('lon', 'f4', ('lon',))
            brightness_temp_var = nc_out.createVariable('Tb', 'f4', ('time', 'lat', 'lon'))

            # Copy the original time, latitude, and longitude values
            nc_out.variables['time'][:] = nc_in.variables['time'][:]
            nc_out.variables['lat'][:] = nc_in.variables['lat'][:]
            nc_out.variables['lon'][:] = nc_in.variables['lon'][:]

            # Set the resampled brightness temperature data
            nc_out.variables['Tb'][:] = resampled_brightness_temp

            # Set the attributes
            time_var.units = nc_in.variables['time'].units
            lat_var.units = nc_in.variables['lat'].units
            lon_var.units = nc_in.variables['lon'].units
            brightness_temp_var.units = nc_in.variables['Tb'].units
            brightness_temp_var.long_name = nc_in.variables['Tb'].long_name

            # Set the new resolution attribute
            nc_out.resolution = new_resolution


# Resample the .nc4 file to 10km resolution
input_file = 'fyp_23-trop_cyclones\DataSources\merg_2022020200_4km-pixel.nc4'
output_file = 'resampled_file.nc4'
scale_factor = 10

resample_nc4_file(input_file, output_file, scale_factor)