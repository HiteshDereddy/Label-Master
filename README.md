# Label Master - Lucario

**Label Master** is a desktop application designed to streamline the image annotation process using AI-assisted object detection. By integrating pre-trained models, the application allows users to quickly label and annotate images while retaining full control for manual adjustments.

## Objective

The goal of this application is to accelerate the image annotation process with minimal user input, ensuring high accuracy and control over the labeling process.

## Tech Stack

- **Programming Language:** Python
- **AI Frameworks:** 
  - PyTorch or TensorFlow for integrating object detection models like YOLO or Faster R-CNN.
- **Image Processing:** 
  - OpenCV for reading, processing, and manipulating images (resizing, loading, etc.).
- **UI Framework:** 
  - Tkinter or PyQt for building a graphical user interface (GUI).
- **Data Export:** 
  - Libraries like JSON, XML, or CSV for structured data export.

## Key Features

1. **AI-Assisted Annotation:** 
   - Automatically predict bounding boxes and labels for objects in images using models like YOLOv5 or Faster R-CNN.
  
2. **Manual Correction Tools:** 
   - Allow users to modify AI-generated annotations by dragging and resizing bounding boxes, changing class labels, or adding new annotations.

3. **Interactive Interface:** 
   - A user-friendly GUI where users can upload images, view predictions, and make adjustments easily.

4. **Data Export:** 
   - Export annotations in JSON, XML, or CSV formats, including image ID, bounding box coordinates, object labels, and metadata.

## Detailed Workflow

1. **Load the Image:**
   - Use OpenCV to load the image and display it in the GUI.
   
2. **Predict Annotations:**
   - Pass the image through a pre-trained object detection model (YOLO or Faster R-CNN) to generate predicted bounding boxes and class labels.
   
3. **Display Predictions:**
   - Show predictions in the GUI and provide options for users to edit them (move/resize boxes, change labels).

4. **Save Annotations:**
   - Implement a save function to export annotations in JSON, XML, or CSV formats, containing bounding box details and their corresponding labels.

## Evaluation Criteria

- **Efficiency:** 
  - Effectiveness in speeding up the labeling process using AI.
  
- **Accuracy:** 
  - Accuracy of object detections and ease of error correction by users.
  
- **Usability:** 
  - User-friendliness of the tool, with clear instructions and minimal technical knowledge required.
  
- **Export Functionality:** 
  - Ease of exporting labeled data for use in other applications (e.g., AI training, analysis).

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HiteshDereddy/LabelMaster.git
   cd LabelMaster
