import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def classify_by_kmeans_color(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape((-1, 3))
    pixels = pixels[np.any(pixels > 15, axis=1)]

    if len(pixels) == 0:
        return "No Egg"

    kmeans = KMeans(n_clusters=1, random_state=42).fit(pixels)
    h, s, v = kmeans.cluster_centers_[0]
    print(f"HSV → H={h:.1f}, S={s:.1f}, V={v:.1f}")

    if s < 30 and v > 150 and 0 < h < 60:
        return "Good Egg"
    elif 18 <= h <= 35 and 40 <= s <= 85 and 140 <= v <= 255:
        return "Bad Egg"
    else:
        return "Uncertain"

def main():
    input_folder = "test images"
    output_folder = "annotated"

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_folder, file)
            img = cv2.imread(path)

            if img is None:
                print(f"Could not read {file}")
                continue

            h, w = img.shape[:2]
            crop = img[h//3:h*2//3, w//3:w*2//3]
            label = classify_by_kmeans_color(crop)

            
            cv2.rectangle(img, (w//3, h//3), (w*2//3, h*2//3), (255, 0, 0), 2)
            cv2.putText(img, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, img)
            print(f"{file} → {label}")

if __name__ == "__main__":
    main()

