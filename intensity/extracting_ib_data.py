# import os
# import pandas as pd

# # Load the CSV data


# # Convert ISO_TIME to a datetime object
# df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

# # Create a directory to store the output files
# output_dir = 'intensity/IB_Extracted'
# os.makedirs(output_dir, exist_ok=True)

# # Iterate through unique storm names in the data
# for storm_name, storm_group in df.groupby('NAME'):
    
#     # Extract and save 'NAME', 'ISO_TIME', 'USA_LAT', 'USA_LON', and 'USA_WIND' columns
#     extracted_data = storm_group[['NAME', 'ISO_TIME', 'USA_LAT', 'USA_LON', 'USA_WIND']]

#     # Create a CSV file for each storm in the specified output directory
#     storm_filename = os.path.join(output_dir, f'{storm_name}.csv')
#     extracted_data.to_csv(storm_filename, index=False)





# Check if any outliers (time logs, where not 3am, 6am, 9am..etc)
import pandas as pd

# Load the CSV data
df = pd.read_excel('intensity/extracted_ib_tracks.xlsx')

# Convert ISO_TIME to a datetime object
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])

# Define the list of times to check
desired_times = ['03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '00:00']

# Iterate through the rows
for index, row in df.iterrows():
    iso_time = row['ISO_TIME']
    storm_name = row['NAME']
    
    # Extract the time portion in the format 'HH:mm'
    time_str = iso_time.strftime('%H:%M')
    
    # Check if the time is not in the desired times list
    if time_str not in desired_times:
        print(f"Storm Name: {storm_name}, ISO_TIME: {iso_time}")
