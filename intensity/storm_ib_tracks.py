import os
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from scipy.interpolate import CubicSpline
import sobel_task1
from PIL import Image
from find_coord import *
from intensity import calculate_intensity

# Create a dictionary to store the data
storm_data = {}

# Specify the directory containing CSV files
directory_path = 'intensity/IB_Extracted'

dav_directory = "DAVs/"
image_directory = "Images/"
counter = 0

my_counter = 0

def calculate_rms_error(known_intensity, calculated_intensity):
    # Calculate the RMS error
    rms_error = np.sqrt(np.mean((known_intensity - calculated_intensity)**2))

    return rms_error

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            storm_name = row['NAME']
            iso_time_str = row['ISO_TIME']
            usa_lat = row['USA_LAT']
            usa_lon = row['USA_LON']
            usa_wind = row['USA_WIND']

            m = str(iso_time_str)
            date_string = m
            date_object = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
            date = date_object.date()
            time = date_object.time()

            # Create a dictionary for the current record
            record = {
                'ISO_TIME': iso_time_str,
                'USA_LAT': usa_lat,
                'USA_LON': usa_lon,
                'USA_WIND': usa_wind
            }

            # Initialize a list for the storm if it doesn't exist
            if storm_name not in storm_data:
                storm_data[storm_name] = {}

            # Initialize a list for the date if it doesn't exist
            if date not in storm_data[storm_name]:
                storm_data[storm_name][date] = []

            # Append the record to the list for the current date
            storm_data[storm_name][date].append(record)

# Create a new data structure to store the data
storm_data_grouped = {}

# Loop through all records of all storms
for storm_name, storm_dates in storm_data.items():
    # Initialize a dictionary for the current storm
    storm_data_grouped[storm_name] = {}
    for date, records in storm_dates.items():
        # Extract relevant data from records
        original_times = []
        original_values = []
        latitudes = []
        longitudes = []
        calculated_intensity = []
        counter += 1
        for record in records:
            iso_time_str = record['ISO_TIME']
            usa_wind = record['USA_WIND']
            usa_lat = record['USA_LAT']
            usa_lon = record['USA_LON']

            # Parse ISO_TIME to get the hour
            iso_time = datetime.strptime(iso_time_str, '%Y-%m-%d %H:%M:%S')
            hour = iso_time.hour

            original_times.append(hour)
            original_values.append(usa_wind)
            latitudes.append(usa_lat)
            longitudes.append(usa_lon)

        # Calculate new interpolated data
        new_times = np.arange(0, 24, 1)
        interpolated_values = np.interp(new_times, original_times, original_values)
        time_points = original_times
        interpolation_time_points = np.arange(0, 24, 1)

        # Cubic spline interpolation for latitudes and longitudes
        spline_lat = CubicSpline(time_points, latitudes, bc_type='natural')
        spline_lon = CubicSpline(time_points, longitudes, bc_type='natural')
        rms = []

        # Store the data under each date, under each storm
        storm_data_grouped[storm_name][date] = {
            'original_times': original_times,
            'original_values': original_values,
            'latitudes': latitudes,
            'longitudes': longitudes,
            'new_times': new_times,
            'interpolated_values': interpolated_values,
            'time_points': time_points,
            'interpolation_time_points': interpolation_time_points,
            'interpolated_latitudes': spline_lat(interpolation_time_points),
            'interpolated_longitudes': spline_lon(interpolation_time_points), 
            'calculated_intensity': calculated_intensity,
            'rms': rms
        }






# Directory containing DAV numpy files
dav_directory = "DAVs/"
image_directory = "Images/"

# Define the start date and time
# we know from ib tracks, first record is from 9th Aug and 3 months = till 1st novemeber
start_datetime = datetime(2021, 8, 1, 0, 0)
end_datetime = datetime(2021, 11, 1, 0, 0)
# Define the number of hours you want to process
num_hours = 24


# Increment start_datetime by a day until it reaches end_datetime
while start_datetime <= end_datetime:
    # Loop through all records of all storms
    flag = False
    for storm_name, storm_dates in storm_data_grouped.items():
        # Loop through the dates for each storm
        # if storm_name == 'ODETTE':
        #     continue
        # if storm_name == 'ROSE':
        #     continue
        # if storm_name == 'TERESA':
        #     continue
        # if storm_name == 'SAM':
        #     continue
        for date, records in storm_dates.items():
            # Convert the date string to a datetime object
            date = str(date)
            date_obj = datetime.strptime(date, '%Y-%m-%d')

            # if date_obj == datetime(2021, 9, 5, 0, 0):
            #     continue  # Skip this specific date

            # if date_obj == datetime(2021, 9, 21, 0, 0):
            #     continue  # Skip this specific date

            # if date_obj == datetime(2021, 9, 22, 0, 0):
            #     continue  # Skip this specific date

            # if date_obj == datetime(2021, 9, 23, 0, 0):
            #     continue  # Skip this specific date

            # if date_obj == datetime(2021, 9, 24, 0, 0):
            #     continue  # Skip this specific date
            # if date_obj == datetime(2021, 9, 25, 0, 0):
            #     continue  # Skip this specific date
            # if date_obj == datetime(2021, 9, 26, 0, 0):
            #     continue  # Skip this specific date
            # if date_obj == datetime(2021, 9, 29, 0, 0):
            #     continue  # Skip this specific date






            # Check if the date matches the current start_datetime
            if start_datetime.date() == date_obj.date():
                radius_km = 36
                dav_values = []
                i = 0
                j =0
                for hour in range(num_hours):
                    #Loads the numpy array of the current hour
                    # Format the hour as a string with leading zeros
                    hour_str = start_datetime.strftime("%Y%m%d%H")

                    # Get the entire date as a string
                    datetime_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Construct the file path for the DAV numpy array
                    file_path = os.path.join(dav_directory, f"merg_{hour_str}_DAV.npy")
                    
                    # Load the numpy array from the file
                    dav_array = np.flipud(np.load(file_path))

                    # Construct the file path for the corresponding image
                    image_path = os.path.join(image_directory, f"merg_{hour_str}.jpg")

                    #Convert image to numpy array
                    image = Image.open(image_path)
                    width, height = image.size
                    image_gray = image.convert('L')
                    gradient_x, gradient_y = sobel_task1.apply_sobel_filter(np.array(image))
                    grad_x = np.reshape(gradient_x, width * height)
                    grad_y = np.reshape(gradient_y, width * height)
                    lon_pts, lat_pts, x_pts, y_pts = numpy_coords(image)

                    interpolated_latitudes = records.get('interpolated_latitudes', [])
                    interpolated_longitudes = records.get('interpolated_longitudes', [])
                        
                    #finds the dav value at the tc centre coordinates
                    while i< len(interpolated_latitudes):
                        xy_coords = []
                        target_latitude = interpolated_latitudes[i]
                        target_longitude = interpolated_longitudes[j]

                        coordinates_within_radius = find_coordinates_around_target(target_latitude, target_longitude, lat_pts, lon_pts, radius_km)

                        for target_latitude, target_longitude in coordinates_within_radius:
                        # Find the indices of the target coordinates in lat_pts and lon_pts
                            target_index = np.where((lat_pts == target_latitude) & (lon_pts == target_longitude))
                            
                            if len(target_index[0]) > 0:
                                target_x = x_pts[target_index[0][0]]
                                target_y = y_pts[target_index[0][0]]
                                xy_coords.append((target_x,target_y))
                            else:
                                print(f"Coordinates ({target_latitude}, {target_longitude}) not found in the image.")

                        total = 0  
                        for x,y in xy_coords:
                            try:
                                total = total + dav_array[int(x)][int(y)]
                            except IndexError:
                                flag = True
                                my_counter +=1
                                break 
                        if flag: 
                            break
                        avg_dav = total/len(xy_coords)
                        dav_values.append(avg_dav)
                        i = i+1
                    if flag:
                        break
                    # Increment the datetime by one hour
                    start_datetime += timedelta(hours=1)
                if flag: 
                    break
                calculated_intensity = []
                for dav in dav_values:
                    calculated_intensity.append(calculate_intensity(dav))
                # print(storm_data_grouped[storm_name])
                # Add calculated_intensity to storm_data_grouped
                storm_data_grouped[storm_name][date_obj.date()]['calculated_intensity'] = calculated_intensity
                known_intensity = interpolated_values  
                storm_data_grouped[storm_name][date_obj.date()]['rms'] = calculate_rms_error(known_intensity, calculated_intensity)
    # Increment start_datetime by one day
    start_datetime += timedelta(days=1)



# # # print(my_counter)

# # Define the start and end dates for the range you want to print
start_date = datetime(2021, 8, 1)
end_date = datetime(2021, 11, 1)




# from datetime import datetime, timedelta

# # Loop through the data grouped by storm
# for storm_name, storm_dates in storm_data_grouped.items():
#     print(f"Storm: {storm_name}")
    
#     # Loop through the dates for each storm
#     for date, data_for_date in storm_dates.items():
#         # Convert the date string to a datetime object
#         date = str(date)
#         date_obj = datetime.strptime(date, '%Y-%m-%d')

#         # Check if the date falls within the desired range
#         if start_date <= date_obj <= end_date:
#             print(f"Date: {date}")
            
#             # Print the data for the current date, including the new variables
#             print(f"original_Times: {data_for_date['original_times']}")
#             print(f"original_Values: {data_for_date['original_values']}")
#             print(f"Latitudes: {data_for_date['latitudes']}")
#             print(f"Longitudes: {data_for_date['longitudes']}")
            
#             # New variables
#             print(f"new_Times: {data_for_date['new_times']}")
#             print(f"interpolated_Values: {data_for_date['interpolated_values']}")
#             print(f"time_points: {data_for_date['time_points']}")
#             print(f"interpolation_time_points: {data_for_date['interpolation_time_points']}")
#             print(f"interpolated_Latitudes: {data_for_date['interpolated_latitudes']}")
#             print(f"interpolated_Longitudes: {data_for_date['interpolated_longitudes']}")
#             print(f"Intensity: {data_for_date['calculated_intensity']}")
#             print(f"RMS: {data_for_date['rms']}")
            
#             # Create og_time_date by combining the date_obj and original_times
#             original_times = data_for_date['original_times']
#             og_time_date = [date_obj + timedelta(hours=int(time)) for time in original_times]
#             print(f"og_time_date: {og_time_date}")



#             # Create intensity_time if 'calculated_intensity' is not empty
#             calculated_intensity = data_for_date['calculated_intensity']
#             if calculated_intensity:
#                 intensity_time = [date_obj.replace(hour=i) for i in range(24)]
#                 print(f"intensity_time: {intensity_time}")
            
#             print("-" * 20)  # Separate data for different dates

#     print("=" * 20)  # Separate data for different storms
##################################################################################################

# BELOW WORKSSSSSS #####


######################################################################################################

import matplotlib.pyplot as plt

# Define the x-axis limits (start_date and end_date)
x_axis_limits = (start_date, end_date)

# Calculate the number of rows and columns based on the number of storms
num_storms = len(storm_data_grouped)
num_rows = num_storms // 2  # Organize into 2 columns
if num_storms % 2 != 0:
    num_rows += 1  # Add an extra row for odd number of storms
num_cols = 2  # 2 columns

# Create a figure with subplots organized into a grid
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))

# Flatten the axs array to make it easier to loop through
axs = axs.flatten()

# Loop through the data grouped by storm
for i, (storm_name, storm_dates) in enumerate(storm_data_grouped.items()):
    ax = axs[i]  # Select the current subplot
    
    # Initialize lists to store blue (original_values) and red (calculated_intensity) data
    blue_x_data = []
    blue_y_data = []
    red_x_data = []
    red_y_data = []
    
    # Loop through the dates for the current storm
    for date, data_for_date in storm_dates.items():
        # Convert the date string to a datetime object
        date = str(date)
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Check if the date falls within the desired range
        if start_date <= date_obj <= end_date:
            # Create og_time_date by combining the date_obj and original_times
            original_times = data_for_date['original_times']
            og_time_date = [date_obj + timedelta(hours=int(time)) for time in original_times]
            
            # Check if calculated_intensity is not empty
            if data_for_date['calculated_intensity']:
                # Create intensity_time
                intensity_time = [date_obj.replace(hour=i) for i in range(24)]
                # Append og_time_date and original_values to the blue data lists
                blue_x_data.extend(og_time_date)
                blue_y_data.extend(data_for_date['original_values'])
                # Append intensity_time and calculated_intensity to the red data lists
                red_x_data.extend(intensity_time)
                red_y_data.extend(data_for_date['calculated_intensity'])
    
    # Plot original_values against og_time_date as a line graph in blue
    if blue_x_data and blue_y_data:
        ax.plot(blue_x_data, blue_y_data, linestyle='-', color='blue', label='Original Values')
    
    # Plot calculated_intensity against intensity_time as a line graph in red
    if red_x_data and red_y_data:
        ax.plot(red_x_data, red_y_data, linestyle='-', color='red', label='Calculated Intensity')
    
    ax.set_title(f"Storm: {storm_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Values")
    
    # Set the same x-axis limits for all subplots
    ax.set_xlim(x_axis_limits)
    
    # Add a legend to the subplot
    ax.legend()

# Remove any empty subplots
for i in range(num_storms, num_rows * num_cols):
    fig.delaxes(axs[i])

# Adjust spacing between subplots for even spacing
plt.subplots_adjust(hspace=0)  # You can adjust the value to control vertical spacing

# Show the plot
plt.show()















































