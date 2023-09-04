import numpy as np
import pandas as pd
import math


def sig(dav):
    a = 1859*(10**-6)
    b = 1437
    sig = 140/(1+ math.exp(a(dav-b))) + 25
    return sig

# data = np.load('dav_values.npy')

# df = pd.DataFrame(data)
# # df.to_excel('output.xlsx', engine='openpyxl', index=False)




# # Replace 'your_file_path.xlsx' with the actual path to your Excel file
# excel_file_path = 'intensity\extracted_ib_tracks.xlsx'


# # Load the Excel file into a pandas DataFrame
# df_ib = pd.read_excel(excel_file_path)

# # Check if 'usa_wind' column exists in the DataFrame
# if 'USA_WIND' in df_ib.columns:
#     usa_wind_column = df_ib['USA_WIND']
#     print(usa_wind_column)

# for column in df:
#     for i in range(len(column)):


        
# print(sig(2314.16)

# print(f"Data from {npy_file_path} has been successfully saved to {csv_file_path}.")

# image = 'intensity\my_plot_ahmed.jpg'

# def numpy_coords(image):
#     w, h = image.size
#     lat_coords = np.full((h*w), 0, dtype='d')
#     lon_coords = np.full((h*w), 0, dtype='d')
#     x_coords = np.full((h*w), 0, dtype='d')
#     y_coords = np.full((h*w), 0, dtype='d')
#     for pixel_ind in range(h*w):
#         x = pixel_ind % w
#         y = pixel_ind // w
#         lat_coords[pixel_ind] = lat_max + (y/h)*(lat_min - lat_max)
#         lon_coords[pixel_ind] = lon_min + (x/w)*(lon_max - lon_min)
#         x_coords[pixel_ind] = x
#         y_coords[pixel_ind] = y
#     return lon_coords, lat_coords, x_coords, y_coords

# print(numpy_coords(image))
