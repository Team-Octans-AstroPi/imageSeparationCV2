import sys
import numpy as np
import cv2
from fastiecm import fastiecm
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
if True:
    img = cv2.imread(path+'/'+filename, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("Could not read the image. Check the filename and path.")


    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    windowUp = np.array([179, 255, 30]) # HSV
    windowDown = np.array([0, 0, 0])

    window = cv2.bitwise_and(img, img, mask = cv2.inRange(img, windowDown, windowUp))

    cloudsUp = np.array([179, 80, 255]) # HSV
    cloudsDown = np.array([0, 0, 185])

    clouds = cv2.bitwise_and(img, img, mask = cv2.inRange(img, cloudsDown, cloudsUp))

    waterUp = np.array([80, 255, 185]) # HSV
    waterDown = np.array([0, 0, 30])

    water = cv2.bitwise_and(img, img, mask = cv2.inRange(img, waterDown, waterUp))

    waterUp = np.array([179, 55, 90]) # HSV
    waterDown = np.array([0, 0, 30])

    water = cv2.bitwise_or(water, cv2.bitwise_and(img, img, mask = cv2.inRange(img, waterDown, waterUp)))

    cv2.imshow('Initial Image', cv2.cvtColor(img, cv2.COLOR_HSV2BGR))
    cv2.imshow('Clouds', cv2.cvtColor(clouds, cv2.COLOR_HSV2BGR))
    cv2.imshow('Water', cv2.cvtColor(water, cv2.COLOR_HSV2BGR))
    cv2.imshow('NDVI Raw', calc_ndvi(cv2.cvtColor(img, cv2.COLOR_HSV2BGR)))
    
    land = img - clouds - water - window # replace with function that checks for < 0
    cv2.imshow('Land', cv2.cvtColor(land, cv2.COLOR_HSV2BGR))
    toBeColorMapped = contrast_stretch(calc_ndvi(cv2.cvtColor(land, cv2.COLOR_HSV2BGR))).astype(np.uint8)
    cv2.imshow('NDVI', toBeColorMapped)
    cv2.imshow('Color Mapped NDVI', cv2.applyColorMap(toBeColorMapped, fastiecm))
    
while True:
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture


