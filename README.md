# Sign Language Recognition using Convolutional Vision Transformers

This project focuses on developing a real-time Indian Sign Language (ISL) recognition system using **Convolutional Vision Transformers (CvT)**. It integrates hand gesture data collection, pre-processing using landmarks, model training, and real-time inference through a GUI application.

---

## 📁 Repository Structure

```
📁 Sign Language Recognition (Root)
├── 📄 README.md
├── 📁 cvt-model
│   ├── 📄 config.json
│   └── 📄 model.safetensors
│
├── 📁 data-collection
│   └── 📄 capture_frames.py
│
├── 📁 inference
│   └── 📄 prototypeV5.py
│
├── 📁 pre-processing
│   ├── 📄 image-augmentation.ipynb
│   └── 📄 image-to-landmark.ipynb
│
├── 📁 technical-report
│   ├── 📄 Conference_Paper_ISL.pdf
│   ├── 📄 FYP_Sign_Language_Recognition_Presentation.pdf
│   └── 📄 FYP_Sign_Language_Recognition_Report.pdf
│
└── 📁 training
    ├── 📄 isl-cvt.ipynb
    ├── 📄 isl-resnet50.ipynb
    └── 📄 isl-vit.ipynb
```


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

## 👥Maintainer
Ankan Dutta <br />
Final Year Project – Indian Sign Language Recognition using
Convolutional Vision Transformers <br />
Department of Computer Science & Engineering <br />
NIT Silchar