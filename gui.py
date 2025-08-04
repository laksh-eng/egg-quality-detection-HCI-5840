import tkinter as tk
from tkinter import filedialog, messagebox
from classify import classify_by_kmeans_color
import cv2
import os
import pandas as pd
from datetime import timedelta

# Create output directories if they don't exist
os.makedirs("imgs", exist_ok=True)
os.makedirs("output_data", exist_ok=True)

#runs egg classification on the selected video.
def process_video_with_gui(input_path, output_path):
    """
    Processes the selected video, classifies eggs using HSV KMeans, and 
    saves the output video with bounding boxes and labels.
    """
    #Loads the video from the selected file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return
    
    #Reads the width, height, and frame rate of the video.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    #Prepares a video writer object to save the processed video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #Updates the GUI label to show the current video being processed.
    label_result["text"] = f"Processing {os.path.basename(input_path)}..."
    root.update()

    #smooth the detection frame to frame
    current_label = None
    consistent_count = 0
    threshold = 10
    final_label = "No Egg"

    egg_data = []
    frame_number = 0
    image_count = 1

    #Loops over every frame in the video.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use the frame as-is (no resizing needed)
        resized = frame.copy()

        # Define ROI at the center of the frame
        box_w, box_h = 400, 400
        x1 = width // 2 - box_w // 2
        y1 = height // 2 - box_h // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        roi = resized[y1:y2, x1:x2]

        # Classify the ROI
        detected_label = classify_by_kmeans_color(roi)

        # Flicker filtering
        if detected_label == current_label:
            consistent_count += 1
        else:
            consistent_count = 0
            current_label = detected_label

        if consistent_count >= threshold:
            final_label = current_label

        # Draw bounding box and classification result
        if final_label != "No Egg":
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(resized, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if final_label == "Good Egg" else (0, 0, 255), 2)
            
            img_filename = f"{image_count}.jpg"
            cv2.imwrite(os.path.join("imgs", img_filename), resized)

            timestamp = str(timedelta(seconds=frame_number / fps))
            egg_data.append({
                "frame": frame_number,
                "time": timestamp,
                "label": final_label,
                "filename": img_filename
            })

            image_count += 1

        # Write and show the frame
        out.write(resized)
        cv2.imshow("Egg Detection", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(egg_data)
    df.to_csv("output_data/egg_log.csv", index=False)
    
    label_result["text"] = "Done. Video saved."
    messagebox.showinfo("Complete", f"Processed and saved output to:\n{output_path}")

def select_video():
    """
    Opens file dialogs to select input video and output save location.
    """
    # Select input video
    filepath = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        filetypes=[("MP4 files", "*.mp4")],
        title="Select Input Video File"
    )
    if not filepath:
        return

    # Select output location
    output_path = filedialog.asksaveasfilename(
        defaultextension=".mp4",
        filetypes=[("MP4 files", "*.mp4")],
        initialfile="gui_detected_output.mp4",
        title="Select Output Video Location"
    )
    if output_path:
        process_video_with_gui(filepath, output_path)

# Create GUI Window
root = tk.Tk()
root.title("Egg Quality Detection (Video)")

label_title = tk.Label(root, text="Egg Quality Detection (Video Classifier)", font=("Arial", 14))
label_title.pack(pady=10)

btn_select_video = tk.Button(root, text="Select and Process Video", command=select_video)
btn_select_video.pack(pady=10)

label_result = tk.Label(root, text="No video processed yet.", font=("Arial", 12))
label_result.pack(pady=10)

btn_quit = tk.Button(root, text="Quit", command=root.destroy)
btn_quit.pack(pady=10)

root.mainloop()

