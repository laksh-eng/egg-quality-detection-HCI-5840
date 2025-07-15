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

# Main Video Processing
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f" Processing video and saving to: {output_path}")
# For flicker protection
    current_label = None
    consistent_count = 0
    threshold = 10  # Number of frames before accepting new label
    final_label = "No Egg"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame (optional)
        resized = cv2.resize(frame, (width, height))

        # Fixed ROI 
        box_w, box_h = 400, 400
        x1 = width // 2 - box_w // 2
        y1 = height // 2 - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        roi = resized[y1:y2, x1:x2]

        detected_label = classify_by_kmeans_color(roi)

        if detected_label == current_label:
            consistent_count += 1
        else:
            consistent_count = 0
            current_label = detected_label

        if consistent_count >= threshold:
            final_label = current_label

        # Only draw bounding box and label if it's not "No Egg"
        if final_label != "No Egg":
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(resized, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if final_label == "Good Egg" else (0, 0, 255), 2)


        

        out.write(resized)
        cv2.imshow("Detection", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved.")

# Run the script
if __name__ == "__main__":
    input_video = "input_video.mp4" 
    output_video = "final_detected_eggs_labeled.mp4"
    process_video(input_video, output_video)











