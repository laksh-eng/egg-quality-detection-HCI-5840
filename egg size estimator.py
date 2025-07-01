import cv2
import numpy as np
import os

def estimate_egg_size(image_path):
    """
    Estimates the egg size by fitting an ellipse to the detected contour.
    Returns the major and minor axes of the ellipse in pixels.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Apply Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Invert threshold if egg is light on dark background
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    #Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {image_path}")
        return None

    #Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    #Check if we can fit an ellipse
    if len(largest_contour) < 5:
        print(f"Not enough points to fit ellipse in {image_path}")
        return None

    #Fit ellipse
    ellipse = cv2.fitEllipse(largest_contour)
    (x, y), (axis1, axis2), angle = ellipse

    #axis1 is major, axis2 is minor
    major_axis = max(axis1, axis2)
    minor_axis = min(axis1, axis2)

    print(f"{os.path.basename(image_path)} → Major Axis: {major_axis:.2f}px, Minor Axis: {minor_axis:.2f}px")
    return major_axis, minor_axis

#Test run on all images in test folder 
if __name__ == "__main__":
    test_folder = "test images"
    for file in os.listdir(test_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_folder, file)
            estimate_egg_size(img_path)

