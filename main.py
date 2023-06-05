import sys
import numpy as np
import cv2
from fastiecm import fastiecm
from octanscm import octanscm
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

def contrast_stretch_original(image):
    in_min = np.percentile(image, 5)
    in_max = np.percentile(image, 95)
    
    out_min= 0.0
    out_max = 255.0
    
    out = image - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min
    
    return out

def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    
    bottom[bottom==0] = 0.01

    ndvi = (b.astype(float) - r) / bottom
    return ndvi

path = "Path here" # change to your own path
filename = "OCTANS_134.jpg" # any IR photo you want to use

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

vegetationUp = np.array([179, 255, 255])
vegetationDown = np.array([30, 20, 55])

vegetation = cv2.bitwise_and(img, img, mask = cv2.inRange(img, vegetationDown, vegetationUp))

cv2.imshow('Initial Image', cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
cv2.imshow('Vegetation Image', cv2.cvtColor(vegetation, cv2.COLOR_HSV2BGR))
cv2.imshow('Clouds', cv2.cvtColor(clouds, cv2.COLOR_HSV2BGR))
cv2.imshow('Water', cv2.cvtColor(water, cv2.COLOR_HSV2BGR))
cv2.imshow('NDVI Raw', calc_ndvi(cv2.cvtColor(img, cv2.COLOR_HSV2BGR)))
raw = contrast_stretch_original(calc_ndvi(cv2.cvtColor(img, cv2.COLOR_HSV2BGR))).astype(np.uint8)
cv2.imshow('Color Mapped Raw NDVI', cv2.applyColorMap(raw, fastiecm))


land = cv2.subtract(img, cv2.add(cv2.add(clouds, water), window))
cv2.imshow('Land', cv2.cvtColor(land, cv2.COLOR_HSV2BGR))
toBeColorMapped = contrast_stretch(calc_ndvi(cv2.cvtColor(vegetation, cv2.COLOR_HSV2BGR))).astype(np.uint8)
cv2.imshow('NDVI', toBeColorMapped)

ndvicm = cv2.applyColorMap(toBeColorMapped, fastiecm)
cv2.imshow('Color Mapped NDVI', cv2.applyColorMap(toBeColorMapped, fastiecm))
cv2.imshow('Color Mapped Octans NDVI', cv2.applyColorMap(toBeColorMapped, octanscm))

"""
    Pixel NDVI Color Count
    This estimates the plant health based on the OCTANSCM color map.
"""

octansNDVI = cv2.applyColorMap(toBeColorMapped, octanscm)

"""
# Slower way - the method below uses numpy, which is heavily optimised and is at least 800% faster than this nested loops
plantPixels = np.count_nonzero(octansNDVI)
plantHealth = [0, 0, 0, 0, 0, 0, 0, 0]

for row in octansNDVI:
    for pixel in row:
        if (pixel != [255, 255, 255]).any():
            plantPixels += 1
            if (pixel == [50, 50, 50]).all():
                plantHealth[0] += 1
            elif (pixel == [120, 120, 120]).all():
                plantHealth[1] += 1
            elif (pixel == [250, 180, 180]).all():
                plantHealth[2] += 1
            elif (pixel == [50, 210, 0]).all():
                plantHealth[3] += 1
            elif (pixel == [5, 223, 247]).all():
                plantHealth[4] += 1
            elif (pixel == [0, 255, 145]).all():
                plantHealth[5] += 1
            elif (pixel == [0, 0, 255]).all():
                plantHealth[6] += 1  
            elif (pixel == [236, 128, 255]).all():
                plantHealth[7] += 1  

#for i in range(len(plantHealth)):
#    plantHealth[i] = plantHealth[i]/plantPixels*100

"""

pixels = octansNDVI.size/3
backgroundPixels = np.count_nonzero((octansNDVI == [255, 255, 255]).all(axis = 2))

vegetationPixels = pixels - backgroundPixels

print("Vegetation pixels:", str(vegetationPixels/pixels*100)+"%", "\t of the image\n")
#print(np.count_nonzero((octansNDVI == [255, 255, 255]).all(axis = 2)))
print("Plant health 1/8: ", str(np.count_nonzero((octansNDVI == [50, 50, 50]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 2/8: ", str(np.count_nonzero((octansNDVI == [120, 120, 120]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 3/8: ", str(np.count_nonzero((octansNDVI == [250, 180, 180]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 4/8: ", str(np.count_nonzero((octansNDVI == [50, 210, 0]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 5/8: ", str(np.count_nonzero((octansNDVI == [5, 223, 247]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 6/8: ", str(np.count_nonzero((octansNDVI == [0, 140, 255]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 7/8: ", str(np.count_nonzero((octansNDVI == [0, 0, 255]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")
print("Plant health 8/8: ", str(np.count_nonzero((octansNDVI == [236, 128, 255]).all(axis = 2))/vegetationPixels*100)+"%", "\t of the plants")

"""
    Dominant Plant Health (not used anymore - replaced by the pixel count above)
    This uses KMeans to compute the dominant plant health
"""

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
kmeans = KMeans(8, max_iter=1024)
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

"""

# wait until q is pressed on the CV2 window
while True:
    if cv2.waitKey(1) == ord('q'):
        break


