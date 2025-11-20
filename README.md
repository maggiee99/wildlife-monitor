# 🐾 Real-Time Wildlife Camera Trap (Streamlit + TensorFlow)

This project is an end-to-end **real-time wildlife classification system** that turns a normal webcam into a smart **camera trap**.

It has two main parts:

1. A **Convolutional Neural Network (CNN)** based on **MobileNetV2** trained to recognize ~99 wildlife species.
2. A **Streamlit web app** that:
   - Opens your webcam
   - Detects motion using OpenCV
   - Crops the moving region
   - Classifies the species in real time
   - Logs predictions for later analysis

---

## 🚩 Problem Statement

Manually monitoring wildlife footage is time-consuming and inefficient.  
The goal is to **automate species detection from camera feeds**, so that interesting wildlife events are flagged and logged automatically.

---

## 🔍 Project Overview

**Core idea:**  
Use a **pre-trained MobileNetV2** model, fine-tune it on a wildlife image dataset (99 species), and deploy it as a **real-time camera trap** with a simple browser UI.

### What the system does

- Loads a **fine-tuned image classifier** and a **label encoder**.
- Uses **background subtraction** to detect motion in webcam frames.
- Crops the motion region and sends it to the model.
- Overlays the **predicted species + confidence** on the live video.
- Shows **training curves (accuracy & loss)**.
- Displays **all available species classes**.
- Lets you **preview the cropped motion** and **export predictions to CSV**.

---

## 🧠 Model & Training (High-Level)

> Training is done offline (e.g., in Google Colab) and the trained model is then used in the Streamlit app.

- **Base model:** `MobileNetV2` (pre-trained on ImageNet, `include_top=False`)
- **Architecture:**
  - Input: `(224, 224, 3)`
  - Data Augmentation: `RandomFlip`, `RandomRotation`
  - Feature extractor: MobileNetV2 (top layers fine-tuned)
  - Classifier head:
    - `GlobalAveragePooling2D`
    - `Dropout`
    - `Dense(256, relu)` with L2 regularization
    - `Dropout`
    - `Dense(num_classes, softmax)`
- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam` with a low learning rate during fine-tuning
- **Training strategy:**
  - Phase A: Freeze base model, train only classifier head
  - Phase B: Unfreeze top layers of MobileNetV2 and fine-tune
  - Use callbacks:
    - `EarlyStopping`
    - `ReduceLROnPlateau`
    - `ModelCheckpoint` (save best model)

**Outputs from training:**

- `saved_models/image_classifier_finetuned.keras`  
- `saved_models/label_encoder.pkl`  
- Optional: training history saved as CSV for plotting curves.

---

## 🌐 Live App (Streamlit)

The app is built with **Streamlit** and **streamlit-webrtc**.

### Main Features

- **Live Camera Trap:**
  - Opens webcam using `streamlit-webrtc`
  - Applies OpenCV **background subtraction (MOG2)** + Gaussian blur
  - Finds the largest motion contour above a threshold
  - Draws a motion bounding box
  - Crops and preprocesses this region
  - Runs the CNN model and overlays prediction

- **Cropped Motion Preview (Sidebar):**
  - Shows the **exact crop** being sent to the model
  - Helps debug motion detection and model behavior

- **Training Curves Tab:**
  - Plots **training vs validation accuracy**
  - Plots **training vs validation loss**

- **Classes / Species Tab:**
  - Lists all species/classes known to the model from `label_encoder.pkl`

- **Prediction Log & CSV Export:**
  - Logs each prediction with **timestamp, species, confidence**
  - Allows download of `predictions_log.csv` from the app

---

## 🧰 Tech Stack

**Languages & Libraries**

- **Python**
- **TensorFlow / Keras**
- **OpenCV** (motion detection, image processing)
- **NumPy, Pandas**
- **Scikit-learn** (LabelEncoder)
- **Matplotlib** (training curves)
- **Streamlit** (web UI)
- **streamlit-webrtc** (webcam video processing)

---

## 📁 Project Structure (Typical)

.
├── app.py                       # Streamlit app (camera trap, plots, tabs)
├── train_classifier.ipynb       # Training notebook / script (MobileNetV2 fine-tuning)
├── requirements.txt             # Python dependencies
├── saved_models/
│   ├── image_classifier_finetuned.keras
│   └── label_encoder.pkl
├── predictions_log.csv          # Auto-created: prediction log from the app
├── training_history.csv         # (Optional) saved accuracy/loss per epoch
└── data/                        # (Optional) local copy of dataset structure
    └── images/
        ├── species_train/
        └── species_validate/


# ⚙️ Setup & Installation (Local)

Clone the repo

git clone https://github.com/<your-username>/wildlife-monitoring-system.git
cd wildlife-monitoring-system


Create & activate a virtual environment (example for Windows PowerShell):

python -m venv venv
venv\Scripts\activate


On Linux / macOS:

python -m venv venv
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Make sure the TensorFlow version in requirements.txt matches the one you used during training.

Place trained artifacts

Copy your trained files into saved_models/:

image_classifier_finetuned.keras

label_encoder.pkl

# ▶️ Run the App
streamlit run app.py


Then open the local URL that Streamlit prints, usually:

http://localhost:8501

From there you can:

Start the Live Camera Trap

View Training Curves

See All Species / Classes

Download Prediction Logs as CSV

# 🚀 Future Improvements

Add top-3 predictions instead of just top-1.

Support uploading images or videos instead of only live webcam.

Add per-class metrics (confusion matrix, per-species accuracy).

Integrate a lighter model (e.g., quantized model, TFLite) for edge devices.

Use active learning: save low-confidence predictions for manual relabeling and retraining.

Deploy on edge hardware (e.g. Raspberry Pi with a camera module).

# 🙏 Acknowledgements

MobileNetV2 pre-trained on ImageNet (TensorFlow / Keras).

Wildlife image dataset sourced from Kaggle (folder-per-species structure).

Streamlit & streamlit-webrtc community for making real-time apps simple.
