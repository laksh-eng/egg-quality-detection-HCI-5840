# Egg Quality Analyzer

A computer vision-based tool that detects and classifies egg quality from videos using HSV color clustering, and estimates egg size from static images using ellipse fitting.

## Overview
This project provides two core tools for automated egg quality detection:
- **Image-based egg size estimation** from photos.
- **Video-based egg classification** using HSV color clustering and a Tkinter GUI.

## Features
- Estimate egg size from still images (`egg_size_estimator.py`)
- Classify eggs as Good/Bad from videos (`classify.py` + `gui.py`)
- Easy GUI interface for video selection and output

## Installation

1. Make sure you have Python 3.8 or later installed.
2. (Optional) Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install required dependencies:
    ```bash
        pip install -r requirements.txt   ```


## How to Use

### GUI Mode (Video Classifier)

1. Run the GUI:
    ```bash
    python gui.py
    ```
2. A window will open:
   - Click **"Select and Process Video"**
   - Choose a `.mp4` video
   - The result will be shown frame by frame
   - Output is saved in `/outputs/` folder (you can customize location)

### Size Estimator (Images Only)

1. Place your test images in the `test images/` folder.
2. Run the size estimator:
    ```bash
    python egg_size_estimator.py
    ```
3. The script fits an ellipse and displays the estimated major/minor axis of each egg.



##  Screenshots 






## Common Issues

| Issue | Solution |
| `No module named _tkinter` | Ensure Python was installed with Tk support (`brew install python-tk`) |
| `File not found: input_video.mp4` | Make sure the video file is inside the `input video/` folder or update the path |
| GUI window doesn't show anything | Press `q` to quit the video window once done |

---

##  Limitations

- Video classification only works when the egg is centered in the frame.
- The model is sensitive to lighting changes.
- HSV thresholds were tuned manually — future improvements can use a trained classifier.
- No real-world size conversion (egg size in pixels only).

---

##  Credits

This project was developed as part of the **HCI 5840** course at Iowa State University.
Author: Lakshmi Jitta  
Professor: [Mr.Chris Harding]


