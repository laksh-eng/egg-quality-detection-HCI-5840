# Developer Guide – Egg Quality Analyzer

This guide is intended to help developers understand, maintain, and extend the Egg Quality Analyzer app.


## Overview

This application uses computer vision techniques (OpenCV + KMeans clustering) to:
- Detect and classify eggs as "Good" or "Bad" from video.
- Estimate egg size from still images using ellipse fitting.

The app is developed for the HCI 5840 course and uses Python 3 with a Tkinter GUI.


## Final Features Implemented

- GUI-based video classifier (`gui.py`)
- Video classification backend (`classify.py`)
- Image-based size estimator (`egg_size_estimator.py`)
- Visual results via bounding box and label
- Optional CSV logging
- User can select input/output paths via GUI


## Installation & Setup

See the [README.md](../README.md) for detailed setup instructions. The developer should ensure:

- Python 3.8+
- Required libraries installed using:
  ```bash
  pip install -r requirements.txt
  ```

No additional admin setup is needed beyond standard installation. No API keys are used.


## Code Architecture and File Roles

```text
egg-quality-analyzer/
classify.py              - Main logic for egg detection from video (uses HSV + KMeans)
gui.py                   - Tkinter GUI interface for user interaction
egg_size_estimator.py    - Ellipse-based egg size estimator for images
test_images/             - Sample images for size estimator
input_video/             - Sample video input folder
outputs/                 - Folder where results (video/images) are saved
requirements.txt         - List of required Python packages
README.md                - User guide
doc/
DEVELOPER_GUIDE.md   - This document
```


## Code Walkthrough

### `gui.py`

- **Role**: Acts as the main user interface.
- **Key Function**: `process_video_with_gui()` handles loading video, running classification, and saving the output.
- Uses OpenCV to process video frames in real-time.
- Displays results frame-by-frame with bounding box + label.

### `classify.py`

- **Role**: Contains the `classify_by_kmeans_color()` function.
- Uses KMeans clustering on HSV pixel values from ROI.
- Outputs "Good Egg", "Bad Egg", or "No Egg" based on color thresholds.

### `egg_size_estimator.py`

- **Role**: Fits an ellipse to the egg contour and extracts major/minor axes.
- Uses thresholding + contour detection.


## User Interaction Flow

1. **User launches `gui.py`**
2. **Selects input video**
3. **Processing occurs** frame-by-frame
4. **Output saved** to selected path
5. Optionally, user runs size estimator separately on still images.


## Known Issues
Severity - Description 
a) Minor   - GUI window might freeze on very large videos 
b) Minor   - No automatic light adjustment — thresholds are sensitive to lighting 
c) Minor    - Assumes centered egg only; off-center eggs may not be classified 

## Suggested Improvements (Future Work)

- Calibrate camera to convert pixels to mm/inches
- Improve detection robustness using ML classifier (e.g., CNN)
- Handle multiple eggs in one frame
- Add toggle in GUI for live webcam support
- Extend GUI to allow batch processing of videos


## Tips for Future Developers

- To add a new detection strategy, modify `classify_by_kmeans_color()` or extend with a new model
- To customize the GUI, edit layout and buttons in `gui.py`
- Test using sample videos under `input_video/` or add your own
- Use `print()` or `cv2.imshow()` for quick debugging


## Maintenance Notes

- Follow PEP8 style guide
- Always test GUI changes in a clean environment
- All outputs should be stored in `outputs/` to avoid clutter


## Contact

If you take over this project, feel free to contact:
- **Lakshmi Jitta** – Original Developer (laksh.eng on GitHub)
