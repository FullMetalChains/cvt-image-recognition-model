import cv2
import torch
import tkinter as tk
import torchvision.transforms as transforms
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, CvtForImageClassification
import torch_tensorrt
import pyttsx3
from collections import deque
import time
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + TRT
model = CvtForImageClassification.from_pretrained('cvt_model').to(device).half()
model = model.to(memory_format=torch.channels_last).eval()

dummy_input = torch.randn(1, 3, 224, 224).to(device).half()
trt_model = torch_tensorrt.compile(
    model,
    inputs=[dummy_input],
    enabled_precisions={torch.half},
    truncate_long_and_double=True
).to(memory_format=torch.channels_last)

# Load processor
processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class mappings (as per your full list)
idx_to_class = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9',
                9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H',
                17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
                25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X',
                33: 'Y', 34: 'Z', 35: 'about', 36: 'above', 37: 'absent', 38: 'accept',
                39: 'afraid', 40: 'agree', 41: 'always', 42: 'around', 43: 'assistance',
                44: 'bad', 45: 'child', 46: 'college', 47: 'doctor', 48: 'father',
                49: 'from', 50: 'ghost', 51: 'group', 52: 'mother', 53: 'old', 54: 'pray',
                55: 'present', 56: 'secondary', 57: 'skin', 58: 'small', 59: 'specific',
                60: 'spouse', 61: 'stand', 62: 'teach', 63: 'there', 64: 'through',
                65: 'today', 66: 'toward', 67: 'up', 68: 'warn', 69: 'wear', 70: 'welcome',
                71: 'which', 72: 'wish', 73: 'with', 74: 'without', 75: 'work', 76: 'you'}

# Voice engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Sign Recognition")

        # Create interface elements
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # Status frame
        status_frame = tk.Frame(root)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(status_frame, text="Status: Waiting for hand...", font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT)
        
        # Confidence threshold slider
        threshold_frame = tk.Frame(root)
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(threshold_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_slider = tk.Scale(threshold_frame, from_=0.0, to=1.0, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=self.threshold_var,
                                  length=200)
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # FPS counter
        self.fps_label = tk.Label(status_frame, text="FPS: 0", font=("Arial", 12))
        self.fps_label.pack(side=tk.RIGHT)

        # Confidence display
        self.confidence_label = tk.Label(root, text="Confidence: 0.00", font=("Arial", 12))
        self.confidence_label.pack(fill=tk.X, padx=10, pady=5)

        # Caption output
        self.caption_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
        self.caption_label.pack(fill=tk.X, padx=10, pady=5)
        
        # Text area for accumulated text
        text_frame = tk.Frame(root)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_area = tk.Text(text_frame, height=5, font=("Arial", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.speak_button = tk.Button(button_frame, text="Speak Text", command=self.speak_text)
        self.speak_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(button_frame, text="Clear Text", command=self.clear_text)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Initialize camera and variables
        self.cap = cv2.VideoCapture(0)
        self.frame_counter = 0
        self.fps_count = 0
        self.fps_start_time = time.time()
        self.prediction_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)
        self.last_confidence = 0.0
        
        # Set up closing protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Calculate FPS
            self.fps_count += 1
            if self.fps_count >= 10:
                elapsed = time.time() - self.fps_start_time
                fps = self.fps_count / elapsed
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.fps_count = 0
                self.fps_start_time = time.time()
            
            # Process every frame for display but only some for prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # For display
            imgtk = ImageTk.PhotoImage(image=frame_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Process every 5th frame for prediction to reduce computation
            self.frame_counter += 1
            if self.frame_counter % 5 == 0:
                # Get prediction and confidence
                pred_class, confidence = self.predict_with_confidence(frame_pil)
                
                # Update confidence label
                self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
                self.last_confidence = confidence
                
                # Threshold check for hand detection
                threshold = self.threshold_var.get()
                if confidence > threshold:
                    self.confidence_history.append(confidence)
                    self.prediction_history.append(pred_class)
                    self.status_label.config(text="Status: Hand detected")
                    
                    # Check if we have a stable prediction
                    if len(self.prediction_history) >= 3:
                        most_common = max(set(self.prediction_history), 
                                         key=self.prediction_history.count)
                        
                        if self.prediction_history.count(most_common) >= 3:
                            self.handle_prediction(most_common)
                            self.prediction_history.clear()
                else:
                    self.status_label.config(text="Status: No hand detected")
                    self.prediction_history.clear()
                    self.confidence_history.clear()
            
            # Draw confidence bar
            if hasattr(self, 'last_confidence'):
                threshold = self.threshold_var.get()
                self.draw_confidence_bar(frame, self.last_confidence, threshold)
        
        # Schedule next frame update
        self.root.after(10, self.update_frame)
    
    def draw_confidence_bar(self, frame, confidence, threshold):
        # Create a visualization to overlay on the frame
        h, w, c = frame.shape
        bar_height = 30
        bar_width = w - 40
        bar_x = 20
        bar_y = h - bar_height - 20
        
        # Draw background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Draw confidence level
        conf_width = int(confidence * bar_width)
        color = (0, 255, 0) if confidence > threshold else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_width, bar_y + bar_height), 
                     color, -1)
        
        # Draw threshold line
        threshold_x = bar_x + int(threshold * bar_width)
        cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), 
                (255, 255, 255), 2)
        
        # Add text
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (bar_x + 10, bar_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def predict_with_confidence(self, image):
        image_tensor = transform(image).unsqueeze(0).to(device).half()
        image_tensor = image_tensor.to(memory_format=torch.channels_last)
        
        with torch.no_grad():
            outputs = trt_model(image_tensor)
            logits = outputs.logits
            
            # Get probabilities with softmax
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Get highest probability and its index
            confidence, pred_idx = torch.max(probs, dim=1)
            
            # Convert to Python values
            pred_idx = pred_idx.item()
            confidence = confidence.item()
            
            # Map index to class
            pred_class = idx_to_class[pred_idx]
            
        return pred_class, confidence

    def handle_prediction(self, cls):
        print(f"Stable prediction: {cls}")

        # Update caption label
        self.caption_label.config(text=f"Prediction: {cls}")

        # Special commands
        if cls == "secondary":
            # Secondary command - speak text
            self.speak_text()
        elif cls == "toward":
            # Backspace command
            current_text = self.text_area.get("1.0", tk.END)[:-1]  # Get all text except the last newline
            if current_text:
                # Remove the last character
                self.text_area.delete("1.0", tk.END)
                self.text_area.insert("1.0", current_text[:-1])
        else:
            # Add the predicted class to the text area
            if cls.isalpha() or cls.isnumeric():
                self.text_area.insert(tk.END, cls + " ")
            else:
                self.text_area.insert(tk.END, cls + " ")
            
            # Scroll to see the latest text
            self.text_area.see(tk.END)

    def speak_text(self):
        text = self.text_area.get("1.0", tk.END).strip()
        if text:
            tts_engine.say(text)
            tts_engine.runAndWait()

    def clear_text(self):
        self.text_area.delete("1.0", tk.END)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
