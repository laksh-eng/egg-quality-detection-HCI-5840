import tkinter as tk
from tkinter import filedialog, messagebox
from classify import classify_by_kmeans_color
import cv2
import os

def process_video_with_gui(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    label_result["text"] = f"Processing {os.path.basename(input_path)}..."
    root.update()

    current_label = None
    consistent_count = 0
    threshold = 10
    final_label = "No Egg"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (width, height))

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

        if final_label != "No Egg":
            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(resized, final_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if final_label == "Good Egg" else (0, 0, 255), 2)

        out.write(resized)
        cv2.imshow("Egg Detection", resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    label_result["text"] = "Done. Video saved."
    messagebox.showinfo("Complete", f"Processed and saved output to:\n{output_path}")

def select_video():
    filepath = filedialog.askopenfilename(
        initialdir="/Users/lakshmi/Desktop",
        filetypes=[("MP4 files", "*.mp4")]
    )
    if filepath:
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

