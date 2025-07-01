import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Egg Classification Logic Using KMeans- Defines a function that takes a part of the image (just the egg) and claasifies it
def classify_by_kmeans_color(region):
    
    #Converts the image from normal color (BGR) to HSV (Hue, Saturation, Value), which is better for analyzing color.
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    #Turns the image into a list of pixels, each with H, S, and V values.
    pixels = hsv.reshape((-1, 3))

    #Removes very dark or black pixels (noise or background).
    pixels = pixels[np.any(pixels > 15, axis=1)]
    #If no valid pixels are left, it returns “No Egg”.
    if len(pixels) == 0:
        return "No Egg"
    

    #Applies KMeans clustering to find the average (main) HSV color in the image.
    kmeans = KMeans(n_clusters=1, random_state=42).fit(pixels)
    h, s, v = kmeans.cluster_centers_[0]
#Prints the dominant color values so you can check them.
    print(f"HSV → H={h:.1f}, S={s:.1f}, V={v:.1f}")

    #Classify based on thresholds
    if v < 100:
        return "No Egg"
    elif s < 30 and v > 185 and 40 < h < 95:
        return "Good Egg"
    elif 18 <= h <= 35 and 40 <= s <= 85 and 150 <= v <= 240:
        return "Bad Egg"
    elif s < 35 and v > 180:
        return "Good Egg"
    elif s > 35:
        return "Bad Egg"
    else:
        return "Uncertain"

#Detect and Crop Egg Using Otsu + Contour
    #This function takes a full image and automatically finds the egg.
def detect_and_crop_egg(image):
    """
    Segment egg using thresholding and crop the bounding box.
    """
    #Converts image to black & white for easier processing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Automatically chooses a brightness level to separate background from egg using Otsu’s method.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Invert if egg is white (background becomes white by default)
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    #Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    #Get the largest contour (assumed to be egg)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y:y+h, x:x+w]

# Main Loop to Process Test Images
def main():
    input_folder = "test images"
    output_folder = "classified"
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read {file}")
            continue

        #Detect and crop the egg
        cropped = detect_and_crop_egg(img)
        if cropped is None:
            print(f"{file} → No egg detected")
            continue

        #Classify cropped region
        label = classify_by_kmeans_color(cropped)

        #Save result image with overlay text
        cv2.putText(img, f"{label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, img)
        print(f"{file} → {label}")

if __name__ == "__main__":
    main()



