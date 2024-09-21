import tkinter as tk
from tkinter import filedialog, Text, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
from transformers import pipeline
import requests
from dataclasses import dataclass
from typing import List
import json
import xml.etree.ElementTree as ET
import pandas as pd

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox

    @classmethod
    def from_dict(cls, detection_dict: dict) -> 'DetectionResult':
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def detect(image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

class AIModelApp:
    def _init_(self, master):
        self.master = master
        master.title("AI Model Image Segmentation")

        # Left frame for buttons and controls
        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Right frame for displaying the image
        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.image_label = tk.Label(self.left_frame, text="Upload an Image:")
        self.image_label.pack()

        self.upload_button = tk.Button(self.left_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.move_boxes_button = tk.Button(self.left_frame, text="Activate Move", command=self.toggle_move_boxes)
        self.move_boxes_button.pack()

        self.prompt_label = tk.Label(self.left_frame, text="Enter labels (comma-separated):")
        self.prompt_label.pack()

        self.prompt_entry = Text(self.left_frame, height=5, width=50)
        self.prompt_entry.pack()

        self.submit_button = tk.Button(self.left_frame, text="Submit", command=self.process_image)
        self.submit_button.pack()

        self.result_label = tk.Label(self.left_frame, text="")
        self.result_label.pack()

        # Initialize export format variable
        self.export_format_var = tk.StringVar(value="JSON")

        self.export_format_label = tk.Label(self.left_frame, text="Select Export Format:")
        self.export_format_label.pack()

        self.json_radio = tk.Radiobutton(self.left_frame, text="JSON", variable=self.export_format_var, value="JSON")
        self.json_radio.pack()

        self.xml_radio = tk.Radiobutton(self.left_frame, text="XML", variable=self.export_format_var, value="XML")
        self.xml_radio.pack()

        self.csv_radio = tk.Radiobutton(self.left_frame, text="CSV", variable=self.export_format_var, value="CSV")
        self.csv_radio.pack()

        self.save_button = tk.Button(self.left_frame, text="Save Image & Export Metadata", command=self.save_image_and_metadata)
        self.save_button.pack()

        # Frame for displaying the image
        self.display_label = tk.Label(self.right_frame)
        self.display_label.pack()

        self.bounding_box_frame = tk.Frame(self.left_frame)
        self.bounding_box_frame.pack()

        # Other necessary initializations
        self.image_path = ""
        self.detections = []
        self.selected_box = None
        self.start_x = None
        self.start_y = None
        self.move_boxes_active = False



    def upload_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", ".jpg;.jpeg;.png;.bmp;*.gif")]
        )
        
        if self.image_path:
            try:
                self.original_image = Image.open(self.image_path)
                self.original_width, self.original_height = self.original_image.size
                
                # Resize and display the image
                self.display_image(self.original_image)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")


    def display_image(self, image: Image.Image):
        max_width, max_height = 1024, 1024
        self.display_width, self.display_height = image.size
        self.x_scale = self.original_width / self.display_width
        self.y_scale = self.original_height / self.display_height
        image.thumbnail((max_width, max_height), Image.BICUBIC)
        self.image_display = ImageTk.PhotoImage(image)
        self.display_label.config(image=self.image_display)
        self.display_label.image = self.image_display
        self.bind_mouse_events()

    def bind_mouse_events(self):
        if self.move_boxes_active:
            self.display_label.bind("<ButtonPress-1>", self.on_button_press)
            self.display_label.bind("<B1-Motion>", self.on_mouse_drag)
            self.display_label.bind("<ButtonRelease-1>", self.on_button_release)
        else:
            self.display_label.unbind("<ButtonPress-1>")
            self.display_label.unbind("<B1-Motion>")
            self.display_label.unbind("<ButtonRelease-1>")

    def on_button_press(self, event):
        if self.move_boxes_active:
            x, y = event.x, event.y
        # Scale the coordinates back to original dimensions
            original_x = int(x * self.x_scale)
            original_y = int(y * self.y_scale)
    
            for detection in self.detections:
                box = detection.box
                if box.xmin <= original_x <= box.xmax and box.ymin <= original_y <= box.ymax:
                    self.selected_box = detection
                    self.start_x, self.start_y = x, y
                    break


    def toggle_move_boxes(self):
        self.move_boxes_active = not self.move_boxes_active
        self.move_boxes_button.config(text="Deactivate Move" if self.move_boxes_active else "Activate Move")
        self.bind_mouse_events()

    def on_mouse_drag(self, event):
        if self.selected_box and self.move_boxes_active:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
        
            self.selected_box.box.xmin += dx
            self.selected_box.box.xmax += dx
            self.selected_box.box.ymin += dy
            self.selected_box.box.ymax += dy
            self.start_x, self.start_y = event.x, event.y
            self.redraw_image()

    def on_button_release(self, event):
        self.selected_box = None

    def redraw_image(self):
        image = load_image(self.image_path)
        annotated_image = self.draw_detections(image, self.detections)
        self.display_image(annotated_image)

    def draw_detections(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        image_cv2 = np.array(image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
    
        for detection in detections:
            box = detection.box
            # Scale the coordinates to display size
            scaled_xmin = int(box.xmin / self.x_scale)
            scaled_ymin = int(box.ymin / self.y_scale)
            scaled_xmax = int(box.xmax / self.x_scale)
            scaled_ymax = int(box.ymax / self.y_scale)
    
            color = np.random.randint(0, 256, size=3).tolist()
            cv2.rectangle(image_cv2, (scaled_xmin, scaled_ymin), (scaled_xmax, scaled_ymax), color, 2)
            cv2.putText(image_cv2, f'{detection.label}: {detection.score:.2f}', 
                        (scaled_xmin, scaled_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
        return Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

    def process_image(self):
        labels_input = self.prompt_entry.get("1.0", tk.END).strip()
        labels = [label.strip() + '.' for label in labels_input.split(',') if label.strip()]

        if self.image_path and labels:
            self.detections = detect(load_image(self.image_path), labels)
            annotated_image = self.draw_detections(load_image(self.image_path), self.detections)
            self.display_image(annotated_image)

            for widget in self.bounding_box_frame.winfo_children():
                widget.destroy()

            for index, detection in enumerate(self.detections):
                button_text = f"Select {detection.label} #{index + 1}"
                button = tk.Button(self.bounding_box_frame, text=button_text, 
                                   command=lambda d=detection: self.select_box(d))
                button.pack(side=tk.TOP, fill=tk.X)

            self.result_label.config(text="Processing complete. You can move the bounding boxes.")
        else:
            messagebox.showerror("Error", "Please upload an image and enter labels.")
        
    def select_box(self, detection: DetectionResult):
        self.selected_box = detection
        messagebox.showinfo("Box Selected", f"Selected box: {detection.label}")

    def save_image_and_metadata(self):
        if not self.image_path or not self.detections:
            messagebox.showerror("Error", "No image or detections to save.")
            return

        final_image_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", ".png"), ("All files", ".*")],
            title="Save the final image"
        )

        if not final_image_path:
            return

        annotated_image = self.draw_detections(load_image(self.image_path), self.detections)
        annotated_image.save(final_image_path)

        metadata = []
        for i, detection in enumerate(self.detections):
            metadata.append({
                "image_id": final_image_path,
                "label": detection.label,
                "score": detection.score,
                "bounding_box": detection.box.xyxy
            })

        export_format = self.export_format_var.get()
        if export_format == "JSON":
            with open(final_image_path.replace('.png', '.json'), 'w') as json_file:
                json.dump(metadata, json_file, indent=4)
        elif export_format == "XML":
            root = ET.Element("detections")
            for item in metadata:
                detection_elem = ET.SubElement(root, "detection")
                ET.SubElement(detection_elem, "image_id").text = item["image_id"]
                ET.SubElement(detection_elem, "label").text = item["label"]
                ET.SubElement(detection_elem, "score").text = str(item["score"])
                bbox_elem = ET.SubElement(detection_elem, "bounding_box")
                bbox_elem.text = str(item["bounding_box"])
            tree = ET.ElementTree(root)
            tree.write(final_image_path.replace('.png', '.xml'))
        elif export_format == "CSV":
            df = pd.DataFrame(metadata)
            df.to_csv(final_image_path.replace('.png', '.csv'), index=False)

        messagebox.showinfo("Success", f"Image and metadata saved successfully as {final_image_path}.")

if _name_ == "_main_":
    root = tk.Tk()
    app = AIModelApp(root)
    root.mainloop()