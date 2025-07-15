import cv2
import numpy as np
from sklearn.cluster import KMeans

#  Egg classification using KMeans HSV 
#This function takes in an image region (ROI) and classifies the egg based on HSV color clustering.
def classify_by_kmeans_color(region):
#Converts the input region from BGR (OpenCV’s default) to HSV (Hue, Saturation, Value) which is more effective for color detection.
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))

    # Ignore dark background pixels
    pixels = pixels[np.any(pixels > 15, axis=1)]
    #Flattens the 2D image into a list of pixels.
    if len(pixels) == 0:
        return "No Egg"
#Clusters the pixel colors to find the dominant HSV colour.
    kmeans = KMeans(n_clusters=1, random_state=42).fit(pixels)
    #Extracts the center HSV value of that cluster
    h, s, v = kmeans.cluster_centers_[0]


    # Prioritizes "Bad Egg" classification first, then "Good Egg", then "Uncertain".
    if h > 50 and s >= 20 and v >=  140: 
        return "Bad Egg"
    elif 44 <= h <= 50 and 20 <= s <= 28 and 140 <= v <= 170:
        return "Good Egg"
    else:
        return "Uncertain"