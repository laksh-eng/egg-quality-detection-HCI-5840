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

    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Increase Gamma 
    gamma = 3  # >1 brightens, <1 darkens
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gray = cv2.LUT(gray, look_up_table)

    #Apply Otsu threshold
    #_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    #cv2.imshow("Threshold1", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 

    #Invert threshold if egg is light on dark background
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    #cv2.imshow("Threshold2", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)         
    thresh = cv2.dilate(thresh, kernel, iterations=1)  

    #cv2.imshow("Threshold3", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 

    # invert it so that contours can be found for largest white area
    thresh = cv2.bitwise_not(thresh)

    #cv2.imshow("Threshold4", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 

    #Find contours
    contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    # Ignore label 0 (background), find the largest component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.uint8(labels == largest_label)
    contours3, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not contours1 or not contours2 or not contours3:
        print(f"No contours found in {image_path}")
        return None

    # Filter contours by area (ignore very small or very large)
    img_area = img.shape[0] * img.shape[1]
    filtered_contours = []
    contours = contours1 + contours2 + contours3
    for c in contours:
        area = cv2.contourArea(c)
        if 0.01 * img_area < area < 0.95* img_area:
            filtered_contours.append(c)
            print(f"Contour area: {area}, Image area: {img_area}")
    if not filtered_contours:
        print(f"No suitable contours found in {image_path}")
        return None

    #Find the largest contour
    largest_contour = max(filtered_contours, key=cv2.contourArea)

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

    # Generate points along the ellipse
    ellipse_points = cv2.ellipse2Poly(
        (int(x), int(y)),
        (int(major_axis/2), int(minor_axis/2)),
        int(angle),
        0, 360, 1
    )
    # Find bounding box
    bbox = cv2.boundingRect(ellipse_points)
    # bbox is (x, y, w, h)

    # CH: Fit ellipsis and draw it in green
    eli_img = img.copy()
    cv2.ellipse(eli_img, ellipse, (0, 255, 0), 2)

    # show for debugging
    #cv2.imshow("Image", img)
    cv2.imshow("Threshold", thresh)
    cv2.imshow("Ellipsis", eli_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    return major_axis, minor_axis, bbox

#Test running on all images in test folder 
if __name__ == "__main__":
    test_folder = "test images"
    for file in os.listdir(test_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(test_folder, file)
            estimate_egg_size(img_path)

