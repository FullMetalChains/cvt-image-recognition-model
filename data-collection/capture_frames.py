import cv2
import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL
import threading
import time
import os
import numpy as np

def select_output_folder():
    folder_selected = filedialog.askdirectory()
    output_folder.set(folder_selected)

def detect_blur(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate the Laplacian variance - higher values mean less blur
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def start_webcam():
    global cap, running
    cap = cv2.VideoCapture(0)
    
    # Try to set higher resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Try to set autofocus if supported
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    running = True
    threading.Thread(target=show_frame).start()

def show_frame():
    global cap, running
    while running:
        ret, frame = cap.read()
        if ret:
            # Display blur score on frame
            blur_score = detect_blur(frame)
            text = f"Blur Score: {blur_score:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show threshold line
            threshold = blur_threshold.get()
            cv2.line(frame, (10, 60), (210, 60), (0, 0, 255), 2)
            cv2.line(frame, (10, 60), (10 + min(int(blur_score), 200), 60), (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {threshold}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def countdown():
    for i in range(5, 0, -1):
        countdown_label.config(text=f"Starting in {i}")
        root.update()
        time.sleep(1)
    countdown_label.config(text="Capturing frames...")
    root.update()
    capture_frames()

def capture_frames():
    global cap, running
    if not cap.isOpened():
        return
    
    output_path = output_folder.get()
    if not output_path:
        countdown_label.config(text="Error: No output folder selected!")
        return
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    frame_count = 0
    saved_count = 0
    rejected_count = 0
    threshold = blur_threshold.get()
    target_count = 50  # We want 50 good images
    
    while saved_count < target_count and running:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Update progress every 10 frames
        if frame_count % 10 == 0:
            countdown_label.config(text=f"Capturing: {saved_count}/{target_count} images")
            root.update()
        
        blur_score = detect_blur(frame)
        
        # Only save if the blur score is above the threshold
        if blur_score > threshold:
            # Save with numeric filename (1.png, 2.png, etc.)
            filename = os.path.join(output_path, f"{saved_count + 1}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
            # Brief pause between captures to allow for movement/scene changes
            time.sleep(0.1)
        else:
            rejected_count += 1
    
    if saved_count >= target_count:
        countdown_label.config(text=f"Capture Complete! Saved {saved_count} images, rejected {rejected_count} blurry images.")
    else:
        countdown_label.config(text=f"Capture stopped! Saved {saved_count}/{target_count} images.")
    root.update()

def start_capture():
    if not output_folder.get():
        countdown_label.config(text="Please select an output folder first!")
        return
    threading.Thread(target=countdown).start()

def stop_capture():
    global running
    running = False
    countdown_label.config(text="Capture stopped by user.")
    root.update()

def close_app():
    global running
    running = False
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    root.quit()

# GUI Setup
root = tk.Tk()
root.title("Webcam Capture with Blur Detection")
root.geometry("500x400")
output_folder = tk.StringVar()

main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill='both', expand=True)

# Output folder selection
folder_frame = tk.Frame(main_frame)
folder_frame.pack(fill='x', pady=10)
tk.Label(folder_frame, text="Output Folder:").pack(side='left')
tk.Entry(folder_frame, textvariable=output_folder, width=30).pack(side='left', padx=5, expand=True, fill='x')
tk.Button(folder_frame, text="Browse", command=select_output_folder).pack(side='right')

# Blur threshold slider
threshold_frame = tk.Frame(main_frame)
threshold_frame.pack(fill='x', pady=10)
tk.Label(threshold_frame, text="Blur Threshold:").pack(anchor='w')
blur_threshold = tk.DoubleVar(value=100.0)  # Default threshold
threshold_slider = Scale(threshold_frame, from_=10, to=500, orient=HORIZONTAL, 
                         variable=blur_threshold, resolution=5)  # Changed from 10 to 5
threshold_slider.pack(fill='x')
tk.Label(threshold_frame, text="Lower = Accepts more images, Higher = Better quality but fewer images").pack(anchor='w')

# Buttons
button_frame = tk.Frame(main_frame)
button_frame.pack(fill='x', pady=20)
tk.Button(button_frame, text="Start Webcam", command=start_webcam, width=15).pack(side='left', padx=5)
tk.Button(button_frame, text="Start Capture", command=start_capture, width=15).pack(side='left', padx=5)
tk.Button(button_frame, text="Stop Capture", command=stop_capture, width=15).pack(side='left', padx=5)
tk.Button(button_frame, text="Exit", command=close_app, width=15).pack(side='right', padx=5)

# Status label
countdown_label = tk.Label(main_frame, text="", font=('Arial', 12))
countdown_label.pack(pady=20)

# Ensure clean exit
root.protocol("WM_DELETE_WINDOW", close_app)

root.mainloop()