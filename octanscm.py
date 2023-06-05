# octanscm - colormap for NDVI

# This colormap, inspired by the Fastie Color Map, uses 8 colors to represent 8 plant health levels.
# It is a custom version that fulfills our ML model needs. To use it at its fullest potential, it must be used on an extracted land image.
# There are 9 colors, since the first one represents the background.

import numpy as np
from numpy import uint8

arr = [[[255, 255, 255]]] + [[[50, 50, 50]]]*31 + [[[120, 120, 120]]]*32 + [[[250, 180, 180]]]*32 + [[[50, 210, 0]]]*32 + [[[5, 223, 247]]]*32 + [[[0, 140, 255]]]*32 + [[[0, 0, 255]]]*32 + [[[236, 128, 255]]]*32

octanscm = np.array(arr, dtype=uint8)