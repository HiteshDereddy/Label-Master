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
from PIL import ImageFont
from PIL import ImageDraw



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
    def __init__(self, master):
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

        self.move_boxes_button = tk.Button(self.left_frame, text="Edit", command=self.toggle_move_boxes)
        self.move_boxes_button.pack()

        self.pen_button = tk.Button(self.left_frame, text="Draw Box", command=self.toggle_pen_mode)
        self.pen_button.pack()

        self.delete_button = tk.Button(self.left_frame, text="Delete Selected Box", command=self.delete_selected_box)
        self.delete_button.pack()

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

        # Bounding box list
        self.bounding_box_listbox = tk.Listbox(self.left_frame, height=10, width=50)
        self.bounding_box_listbox.pack()

        self.bounding_box_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # Other necessary initializations
        self.drawing_active = False
        self.start_x = None
        self.start_y = None
        self.current_box = None
        self.image_path = ""
        self.detections = []
        self.selected_box = None
        self.start_x = None
        self.start_y = None
        self.move_boxes_active = False
        self.resizing_active = False
        self.resizing_box = False

    def toggle_pen_mode(self):
        self.drawing_active = not self.drawing_active
        self.pen_button.config(text="Stop Drawing" if self.drawing_active else "Draw Box")
        self.bind_mouse_events()

    def delete_selected_box(self):
        """Delete the selected bounding box and update the listbox and detections."""
        try:
            selected_index = self.bounding_box_listbox.curselection()[0]
            del self.detections[selected_index]  # Remove the detection
            self.bounding_box_listbox.delete(selected_index)  # Remove from listbox
            
            # Redraw image without the deleted box
            annotated_image = self.draw_detections(load_image(self.image_path), self.detections)
            self.display_image(annotated_image)

            self.result_label.config(text="Box deleted successfully.")
        except IndexError:
            messagebox.showerror("Error", "No bounding box selected to delete.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
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
        if self.drawing_active:
            self.start_x = event.x
            self.start_y = event.y
            self.current_box = BoundingBox(xmin=int(event.x * self.x_scale),
                                            ymin=int(event.y * self.y_scale),
                                            xmax=int(event.x * self.x_scale),
                                            ymax=int(event.y * self.y_scale))
        elif self.move_boxes_active:
            x, y = event.x, event.y
            # Scale the coordinates back to original dimensions
            original_x = int(x * self.x_scale)
            original_y = int(y * self.y_scale)

            for detection in self.detections:
                box = detection.box
                # Check if click is near edges for resizing or inside the box for moving
                if self.is_near_edge(original_x, original_y, box):
                    self.resizing_active = True
                    self.selected_box = detection
                    self.start_x, self.start_y = x, y
                    break
                elif box.xmin <= original_x <= box.xmax and box.ymin <= original_y <= box.ymax:
                    self.selected_box = detection
                    self.start_x, self.start_y = x, y
                    break

    def is_near_edge(self, x, y, box, margin=5):
        """Check if the coordinates are near the edge of the bounding box for resizing."""
        return (abs(x - box.xmax) <= margin or abs(x - box.xmin) <= margin) or \
               (abs(y - box.ymax) <= margin or abs(y - box.ymin) <= margin)

    def toggle_move_boxes(self):
        self.move_boxes_active = not self.move_boxes_active
        self.move_boxes_button.config(text="Unedit" if self.move_boxes_active else "Edit")
        self.bind_mouse_events()

    def on_mouse_drag(self, event):
        if self.drawing_active and self.current_box:
            # Update the current box while dragging
            self.current_box.xmax = int(event.x * self.x_scale)
            self.current_box.ymax = int(event.y * self.y_scale)
            self.redraw_image()
            
        elif self.selected_box and self.move_boxes_active:
            dx = event.x - self.start_x
            dy = event.y - self.start_y
    
            if self.resizing_active:
                # Resize the bounding box
                self.selected_box.box.xmax += int(dx * self.x_scale)
                self.selected_box.box.ymax += int(dy * self.y_scale)
            else:
                # Move the bounding box
                self.selected_box.box.xmin += int(dx * self.x_scale)
                self.selected_box.box.xmax += int(dx * self.x_scale)
                self.selected_box.box.ymin += int(dy * self.y_scale)
                self.selected_box.box.ymax += int(dy * self.y_scale)
    
            # Update the start position for the next drag
            self.start_x, self.start_y = event.x, event.y
            self.redraw_image()  # Redraw the image to reflect changes
    
    def on_button_release(self, event):
        if self.drawing_active and self.current_box:
            label = simpledialog.askstring("Label", "Enter label for the bounding box:")
            if label:
                # Finalize the bounding box with the entered label
                new_detection = DetectionResult(score=1.0, label=label, box=self.current_box)
                self.detections.append(new_detection)
                
                # Update the listbox with the correct numbering
                self.bounding_box_listbox.insert(tk.END, f"{label}. #{len(self.detections)}")
                
                # Do not reset current_box here; keep the existing boxes
                self.current_box = None  # Reset only the current box for new drawings
                self.redraw_image()
            else:
                messagebox.showwarning("Warning", "Label cannot be empty.")
        else:
            # Clear selection on mouse release
            self.selected_box = None
            self.resizing_active = False

    
    def redraw_image(self):
        # Create a copy of the original image to draw on
        annotated_image = self.original_image.copy()
        image_cv2 = np.array(annotated_image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
        
        # Draw existing boxes
        for detection in self.detections:
            box = detection.box
            cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
            cv2.putText(image_cv2, detection.label, (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw the current box if in drawing mode
        if self.current_box:
            cv2.rectangle(image_cv2, (self.current_box.xmin, self.current_box.ymin), (self.current_box.xmax, self.current_box.ymax), (255, 0, 0), 2)
        
        # Convert back to RGB and then to PhotoImage
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(image_cv2)
        self.tk_image = ImageTk.PhotoImage(annotated_image)
        
        # Update the display with the annotated image
        self.display_label.configure(image=self.tk_image)
        self.display_label.image = self.tk_image  # Keep a reference




    def draw_detections(self, image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
        image_cv2 = np.array(image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
        
        for detection in detections:
            label = detection.label  # Use only the label without numbering
            box = detection.box
            
            cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), (255, 0, 0), 2)
            cv2.putText(image_cv2, label, (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_cv2)


    def process_image(self):
        labels_input = self.prompt_entry.get("1.0", tk.END).strip()
        labels = [label.strip() + '.' for label in labels_input.split(',') if label.strip()]

        if self.image_path and labels:
            self.detections = detect(load_image(self.image_path), labels)
            annotated_image = self.draw_detections(load_image(self.image_path), self.detections)
            self.display_image(annotated_image)

            # Clear the Listbox and add new entries
            self.bounding_box_listbox.delete(0, tk.END)

            for index, detection in enumerate(self.detections):
                listbox_entry = f"{detection.label} #{index + 1}"
                self.bounding_box_listbox.insert(tk.END, listbox_entry)

            self.result_label.config(text="Processing complete. You can move the bounding boxes.")
        else:
            messagebox.showerror("Error", "Please upload an image and enter labels.")

    def on_listbox_select(self, event):
        """Triggered when the user selects an item from the Listbox."""
        try:
            # Get the index of the selected item
            selected_index = self.bounding_box_listbox.curselection()[0]
            self.select_box(self.detections[selected_index])
        except IndexError:
            pass  # No selection made

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

if __name__ == "__main__":
    root = tk.Tk()
    app = AIModelApp(root)
    root.mainloop()
