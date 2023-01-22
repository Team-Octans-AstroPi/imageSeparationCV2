import sys
import numpy as np
import cv2
from fastiecm import fastiecm
from sklearn.cluster import MiniBatchKMeans as KMeans # Either MiniBatchKMeans (faster) or KMeans (more accurate)
import time
import os

def contrast_stretch(im):
    in_min = 0.0
    in_max = min(0.4345989304812834, np.percentile(im, 95))

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    
    bottom[bottom==0] = 0.01

    ndvi = (b.astype(float) - r) / bottom
    return ndvi

path = "../IR-photos" # change to your own path
filename = "photo_01086_51846002189_o.jpg" # any IR photo you want to use

img = cv2.imread(path+'/'+filename, cv2.IMREAD_COLOR)
if img is None:
    sys.exit("Could not read the image. Check the filename and path.")


img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

windowUp = np.array([179, 255, 30])
windowDown = np.array([0, 0, 0])

window = cv2.bitwise_and(img, img, mask = cv2.inRange(img, windowDown, windowUp))

cloudsUp = np.array([179, 80, 255])
cloudsDown = np.array([0, 0, 185])

clouds = cv2.bitwise_and(img, img, mask = cv2.inRange(img, cloudsDown, cloudsUp))

waterUp = np.array([80, 255, 185])
waterDown = np.array([0, 0, 30])

water = cv2.bitwise_and(img, img, mask = cv2.inRange(img, waterDown, waterUp))

waterUp = np.array([179, 55, 90])
waterDown = np.array([0, 0, 30])

water = cv2.bitwise_or(water, cv2.bitwise_and(img, img, mask = cv2.inRange(img, waterDown, waterUp)))

#cv2.imshow('Initial Image', cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
#cv2.imshow('Clouds', cv2.cvtColor(clouds, cv2.COLOR_HSV2BGR))
#cv2.imshow('Water', cv2.cvtColor(water, cv2.COLOR_HSV2BGR))
#cv2.imshow('NDVI Raw', calc_ndvi(cv2.cvtColor(img, cv2.COLOR_HSV2BGR)))

land = cv2.subtract(img, cv2.add(cv2.add(clouds, water), window))
#cv2.imshow('Land', cv2.cvtColor(land, cv2.COLOR_HSV2BGR))
toBeColorMapped = contrast_stretch(calc_ndvi(cv2.cvtColor(land, cv2.COLOR_HSV2BGR))).astype(np.uint8)
#cv2.imshow('NDVI', toBeColorMapped)

ndvicm = cv2.applyColorMap(toBeColorMapped, fastiecm)
cv2.imshow('Color Mapped NDVI', cv2.applyColorMap(toBeColorMapped, fastiecm))

"""
    Dominant Plant Health
    This uses KMeans to compute the most 
"""

# scale down the original image to make KMeans faster
scale = 0.25 
w = int(toBeColorMapped.shape[1] * scale)
h = int(toBeColorMapped.shape[0] * scale)
print("Kmeans rescaled image size:", (w, h))
toBeColorMapped = cv2.resize(toBeColorMapped, (w, h), interpolation = cv2.INTER_LINEAR)

# Scikit-learn KMeans Cluster implementation
pixels = np.uint8(toBeColorMapped.reshape((-1, 1))) # convert image to 1D array of elements (element = array of one value)
print("Finding KMeans clusters...\nFound clusters in", end=" ")
ms = time.time()*1000.0
kmeans = KMeans(4, max_iter=1024)
kmeans.fit(pixels)
print(time.time()*1000.0-ms, "ms") # print time taken to compute KMeans, useful when searching for the best max_iter, scale, and KMeans type.

# print data
print("Average plant health: ", np.average(ndvicm))
print("Median plant health: ", np.median(ndvicm))
print("85th Percentile plant health: ", np.percentile(toBeColorMapped, 85))
print("K-Means Clustering:\n", kmeans.cluster_centers_)

# show computed dominant colors in OpenCV windows
for color in range(0, np.shape(kmeans.cluster_centers_)[0]):
    array = [[kmeans.cluster_centers_[color][0] for _ in range(200)] for _ in range(200)]
    array = np.array(array).astype(np.uint8)

    cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Dominant plant health "+str(color), cv2.applyColorMap(array, fastiecm))


# wait until q is pressed on the CV2 window
while True:
    if cv2.waitKey(1) == ord('q'):
        break


