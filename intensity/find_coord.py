
import numpy as np
import sobel_task1
from PIL import Image
import matplotlib.pyplot as plt



def numpy_coords(image):
        w, h = image.size
        lat_coords = np.full((h*w), 0, dtype='d')
        lon_coords = np.full((h*w), 0, dtype='d')
        x_coords = np.full((h*w), 0, dtype='d')
        y_coords = np.full((h*w), 0, dtype='d')
        for pixel_ind in range(h*w):
            x = pixel_ind % w
            y = pixel_ind // w
            lat_coords[pixel_ind] = lat_max + (y/h)*(lat_min - lat_max)
            lon_coords[pixel_ind] = lon_min + (x/w)*(lon_max - lon_min)
            x_coords[pixel_ind] = x
            y_coords[pixel_ind] = y
        return lon_coords, lat_coords, x_coords, y_coords


def find_coordinates_around_target(target_latitude, target_longitude, lat_coords, lon_coords, radius_km):
    # Convert radius from kilometers to degrees (approximately)
    degrees_per_km = 1 / 111.32  # Rough approximation
    radius_degrees = radius_km * degrees_per_km

    # Calculate the squared distance between each coordinate and the target
    squared_distances = (lat_coords - target_latitude)**2 + (lon_coords - target_longitude)**2

    # Find the indices of coordinates within the specified radius
    within_radius_indices = np.where(squared_distances <= radius_degrees**2)

    # Extract the coordinates within the radius
    coordinates_within_radius = [(lat_coords[i], lon_coords[i]) for i in within_radius_indices[0]]

    return coordinates_within_radius




# Define the North Atlantic region (in degrees)
lon_min, lon_max = -120, 0
lat_min, lat_max = -5, 60


image = Image.open('another_plot.jpg')
width, height = image.size 
image_gray = image.convert('L')
gradient_x, gradient_y = sobel_task1.apply_sobel_filter(np.array(image))
grad_x = np.reshape(gradient_x, width * height)
grad_y = np.reshape(gradient_y, width * height)

lon_pts, lat_pts, x_pts, y_pts = numpy_coords(image)


target_latitude = 17.3
target_longitude = -66.1
radius_km = 10  # Adjust the radius as needed

coordinates_within_radius = find_coordinates_around_target(target_latitude, target_longitude, lat_pts, lon_pts, radius_km)

print(coordinates_within_radius)

target_latitude_1 = coordinates_within_radius[0][0]
target_longitude_2 = coordinates_within_radius[0][1]

target_index = np.where((lat_pts == target_latitude_1) & (lon_pts == target_longitude_2))
if len(target_index[0]) > 0:
    target_x = x_pts[target_index[0][0]]
    target_y = y_pts[target_index[0][0]]
    print(f"For Latitude: {target_latitude}, Longitude: {target_longitude}")
    print(f"Corresponding x point: {target_x}, y point: {target_y}")
else:
    print("Coordinates not found in the image.")