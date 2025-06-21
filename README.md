# Sign Language Recognition using Convolutional Vision Transformers

This project focuses on developing a real-time Indian Sign Language (ISL) recognition system using **Convolutional Vision Transformers (CvT)**. It integrates hand gesture data collection, pre-processing using landmarks, model training, and real-time inference through a GUI application.

---

## 📁 Repository Structure

.
├── README.md
├── cvt-model/ # Trained CvT model files (config and weights)
│ ├── config.json
│ └── model.safetensors
│
├── data-collection/ # GUI tool to collect sign gesture data
│ └── capture_frames.py
│
├── inference/ # Real-time GUI-based inference application
│ └── prototypeV5.py
│
├── pre-processing/ # Data augmentation and hand landmark extraction
│ ├── image-augmentation.ipynb
│ └── image-to-landmark.ipynb
│
├── technical-report/ # Publications and final reports
│ ├── Conference_Paper_ISL.pdf
│ ├── FYP_Sign_Language_Recognition_Presentation.pdf
│ └── FYP_Sign_Language_Recognition_Report.pdf
│
└── training/ # Training pipelines for CvT, ViT, ResNet
├── isl-cvt.ipynb
├── isl-resnet50.ipynb
└── isl-vit.ipynb


---

## 🚀 Modules Overview

### 🔹 cvt-model
Contains the trained CvT model used during inference. Includes the `config.json` and the model weights in `.safetensors` format.

### 🔹 data-collection
A Python-based GUI tool using OpenCV and Tkinter to capture labeled hand gesture images via webcam, for building a custom ISL dataset.

### 🔹 pre-processing
Notebooks to:
- Augment the collected dataset (flip, rotation, zoom, etc.)
- Extract 2D hand landmarks using MediaPipe, transforming raw images into structured input for model training.

### 🔹 training
Model training pipelines for:
- **CvT**: Our primary architecture, optimized for local spatial bias with transformer scalability.
- **ViT**: A baseline Vision Transformer model.
- **ResNet50**: A CNN-based baseline for comparison.

Each notebook includes data loading, training, evaluation, and model saving.

### 🔹 inference
A real-time Tkinter + OpenCV app that:
- Loads a trained model
- Captures live webcam input
- Predicts sign classes
- Displays results on the GUI

### 🔹 technical-report
Includes:
- 📄 Final thesis report
- 🎓 Conference paper submission
- 📊 Project presentation slides

---

## 🧠 Core Technologies

- **Convolutional Vision Transformers (CvT)**
- **MediaPipe Hands** for landmark extraction
- **OpenCV & Tkinter** for GUI applications
- **PyTorch** for model training and inference
- **Jupyter Notebooks** for modular experimentation

---

👥 Authors
Ankan Dutta, Spandan Priyam Chetia
Final Year Project – Indian Sign Language Recognition
Department of Computer Science & Engineering
NIT Silchar