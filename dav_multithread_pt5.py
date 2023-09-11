import numpy as np
from scipy.signal import convolve2d
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing
import sobel_task1
import numba
from numba import vectorize, njit, prange, cuda, jit

"""Calculate distance by pixel resolution"""
@njit(parallel=True)
def dist_pixel(x1, y1, x2, y2):
    res = 4
    return np.sqrt(((res * (x1 - x2)) ** 2) + ((res * (y1 - y2)) ** 2))

@njit(parallel=True)
def calculate_DAV_Numpy(gradient_x, gradient_y, radial_x, radial_y):
    # Calculate the dot product
    dot_product = gradient_x*radial_x + gradient_y*radial_y
    grad_mag = np.sqrt((gradient_x**2) + (gradient_y**2))
    rad_mag = np.sqrt((radial_x**2) + (radial_y**2))
    ind = np.where(grad_mag > 0)

    # Clip the ratios to be in range between -1 and 1
    ratios = dot_product[ind] / (grad_mag[ind] * rad_mag[ind])
    ratios = np.where(ratios >= -1, ratios, -1)
    ratios = np.where(ratios <= 1, ratios, 1)

    # Calculate the deviation angle
    deviation_angle = np.arccos(ratios)
    deviations = np.degrees(deviation_angle)
    angles = np.where(deviations <= 90, deviations, deviations - 180)

    # Calculate variance if there are deviation angles
    if angles.size > 0:
        variance = np.nanvar(angles)
    else:
        variance = 0

    return variance

@njit(parallel=True)
def numpy_coords(h, w):
    x_coords = np.full((h*w), 0, dtype='d')
    y_coords = np.full((h*w), 0, dtype='d')
    for pixel_ind in prange(h*w):
        x = pixel_ind % w
        y = pixel_ind // w
        x_coords[pixel_ind] = x
        y_coords[pixel_ind] = y
    return x_coords, y_coords

@njit(parallel=True)
def get_variance(height, width, radial_dist, x_pts, y_pts, grad_x, grad_y):
    dav_array = np.zeros((height, width), dtype='d')
    for pixel_ind in prange(height * width):
        x = pixel_ind % width
        y = pixel_ind // width
        radial_x = x - x_pts
        radial_y = y_pts - y
        ind = np.where((0 < dist_pixel(x, y, x_pts, y_pts)) &
                        (dist_pixel(x, y, x_pts, y_pts) <= radial_dist))
        variance = calculate_DAV_Numpy(grad_x[ind], grad_y[ind],
                                                radial_x[ind], radial_y[ind])
        dav_array[y, x] = variance
    return dav_array


def run_algorithm(filename):

    dataname = filename[:filename.find('_', 5)]
    imgname = "Images/" + dataname + ".jpg"

    # Load the image and convert it to a numpy array. This is assuming that image has already been made
    image = Image.open(imgname)
    width, height = image.size
    image_gray = image.convert('L')
    gradient_x, gradient_y = sobel_task1.apply_sobel_filter(np.array(image))
    grad_x = np.reshape(gradient_x, width * height)
    grad_y = np.reshape(gradient_y, width * height)


    # With different radial distances, calculate DAV
    width, height = image.size
    x_pts, y_pts = numpy_coords(height, width)

    # Now Mapping deviation-angle variances
    dav_array = get_variance(height, width, 250, x_pts, y_pts, grad_x, grad_y)

    # get the dav values masked out where brightness temperature is too high
    #final_DAVs = np.where(((image_gray / 255) * (max_temp - min_temp) + min_temp) <= temp_threshold, dav_array, 0)
    davname = "DAVs/" + dataname + "_DAV.npy"
    np.save(davname, dav_array)


def use_data(start, end):
    for i in range(start, end):
        line = allFiles[i].strip()
        dataname = line[line.find('merg_'):]
        run_algorithm(dataname)

# Define the North Atlantic region (in degrees)
lon_min, lon_max = -120, 0
lat_min, lat_max = -5, 60
dataList = open('netcdfList.txt')
allFiles = dataList.readlines()
dataList.close()
use_data(1472, 1840)

