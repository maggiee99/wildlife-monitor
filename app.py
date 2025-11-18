import os
import time
import csv
from datetime import datetime

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
import cv2
import av
import pandas as pd

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ===================== BASIC CONFIG =====================
st.set_page_config(
    page_title="Live Wildlife Camera Trap",
    page_icon="🤖",
    layout="wide",
)

st.title("Live Wildlife Camera Trap 🤖")

# -------------------- Paths --------------------
MODEL_PATH = "saved_models/image_classifier_finetuned.h5"
ENCODER_PATH = "saved_models/label_encoder.pkl"

HISTORY_NPY = "saved_models/training_history.npy"
HISTORY_PKL = "saved_models/training_history.pkl"

PREDICTIONS_CSV = "predictions_log.csv"
PREDICTION_LOG_PATH = "predictions_log.csv"

# -------------------- Constants --------------------
IMG_SIZE = (224, 224)

# ===================== SIDEBAR CONTROLS =====================
st.sidebar.header("Detection Settings")

MOTION_THRESHOLD = st.sidebar.slider(
    "Motion area threshold (larger = less sensitive)",
    min_value=100,
    max_value=5000,
    value=500,
    step=100,
)

PREDICTION_COOLDOWN = st.sidebar.slider(
    "Prediction cooldown (seconds)",
    min_value=0.1,
    max_value=2.0,
    value=0.5,
    step=0.1,
)

MIN_CONFIDENCE = st.sidebar.slider(
    "Min confidence to trust prediction",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
)

SHOW_CROP_PREVIEW = st.sidebar.checkbox(
    "Show cropped motion preview", value=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Predictions are also logged to `predictions_log.csv`")

if os.path.exists(PREDICTION_LOG_PATH):
    with open(PREDICTION_LOG_PATH, "rb") as f:
        st.sidebar.download_button(
            label="⬇️ Download prediction log",
            data=f,
            file_name="predictions_log.csv",
            mime="text/csv",
        )
else:
    st.sidebar.caption("No predictions logged yet.")
# ===================== MODEL LOADING =====================

@st.cache_resource
def load_model_and_encoder():
    model = keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


model, label_encoder = load_model_and_encoder()

if model is None or label_encoder is None:
    st.error(
        f"❌ Could not load model or label encoder. Check that `{MODEL_PATH}` "
        f"and `{ENCODER_PATH}` exist."
    )
else:
    st.success(f"✅ Model Loaded: {len(label_encoder.classes_)} species")


# ===================== VIDEO PROCESSOR =====================

class CameraTrapProcessor(VideoTransformerBase):
    def __init__(self):
        # Model & encoder already loaded globally and cached
        self.model = model
        self.label_encoder = label_encoder

        # Motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # State
        self.last_prediction_time = 0.0
        self.last_species_name = "Waiting for motion..."
        self.last_confidence = 0.0

        # For preview in sidebar
        self.last_cropped_frame = None

        # CSV logging
        self.csv_path = PREDICTIONS_CSV
        if not os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "species", "confidence"])
            except Exception as e:
                print(f"[WARN] Could not create CSV log file: {e}")

    def _preprocess_image(self, cropped_frame: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(cropped_frame, IMG_SIZE)
        img_resized = img_resized.astype(np.float32) / 255.0
        return np.expand_dims(img_resized, axis=0)

    def _log_prediction(self):
        """Append last prediction to CSV."""
        try:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        self.last_species_name,
                        float(self.last_confidence),
                    ]
                )
        except Exception as e:
            print(f"[WARN] Failed to write prediction to CSV: {e}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated_frame = img.copy()

        # ---------- Motion Detection ----------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        fg_mask = self.bg_subtractor.apply(gray_blurred)
        fg_mask = cv2.erode(fg_mask, None, iterations=1)
        fg_mask = cv2.dilate(fg_mask, None, iterations=4)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = None
        max_area = 0
        frame_h, frame_w = img.shape[:2]
        max_allowed_area = 0.9 * frame_h * frame_w

        for c in contours:
            area = cv2.contourArea(c)
            if MOTION_THRESHOLD < area < max_allowed_area and area > max_area:
                max_area = area
                largest_contour = c

        # ---------- Classification ----------
        if largest_contour is not None and self.model is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(
                annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2
            )

            current_time = time.time()
            if (current_time - self.last_prediction_time) > PREDICTION_COOLDOWN:
                self.last_prediction_time = current_time

                y1, y2 = max(0, y), min(frame_h, y + h)
                x1, x2 = max(0, x), min(frame_w, x + w)

                cropped_motion = img[y1:y2, x1:x2]
                if cropped_motion.size > 0:
                    self.last_cropped_frame = cropped_motion.copy()

                    try:
                        processed = self._preprocess_image(cropped_motion)
                        preds = self.model.predict(processed, verbose=0)[0]
                        pred_index = int(np.argmax(preds))
                        self.last_confidence = float(np.max(preds))

                        if self.last_confidence < MIN_CONFIDENCE:
                            self.last_species_name = "Unknown / Low confidence"
                        else:
                            raw_name = self.label_encoder.inverse_transform(
                                [pred_index]
                            )[0]
                            self.last_species_name = (
                                raw_name.replace("_", " ").title()
                            )

                        # Log to CSV
                        self._log_prediction()

                    except Exception as e:
                        print(f"[ERROR] Prediction failed: {e}")
                        self.last_species_name = "Error"
                        self.last_confidence = 0.0

        # ---------- Draw Result Text ----------
        text = f"{self.last_species_name} ({self.last_confidence*100:.0f}%)"
        cv2.putText(
            annotated_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# ===================== TABS =====================

tab_live, tab_curves, tab_species = st.tabs(
    ["📹 Live Camera Trap", "📈 Training Curves", "📚 Species Classes"]
)

# ---------- TAB 1: Live Camera Trap ----------
with tab_live:
    st.subheader("Live Wildlife Detection")
    st.caption("Click START below to begin the webcam stream.")

    ctx = webrtc_streamer(
        key="camera-trap",
        video_processor_factory=CameraTrapProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    # Show cropped motion preview in sidebar
    if SHOW_CROP_PREVIEW and ctx and ctx.video_processor:
        vp = ctx.video_processor
        if vp.last_cropped_frame is not None:
            crop_rgb = cv2.cvtColor(vp.last_cropped_frame, cv2.COLOR_BGR2RGB)
            st.sidebar.image(
                crop_rgb,
                caption="Cropped motion fed to model",
                use_column_width=True,
            )
        else:
            st.sidebar.info("No motion crop captured yet.")


# ---------- TAB 2: Training Curves ----------
with tab_curves:
    st.subheader("Training Accuracy & Loss")

    history = None
    if os.path.exists(HISTORY_NPY):
        try:
            history = np.load(HISTORY_NPY, allow_pickle=True).item()
        except Exception as e:
            st.error(f"Failed to load {HISTORY_NPY}: {e}")
    elif os.path.exists(HISTORY_PKL):
        import pickle

        try:
            with open(HISTORY_PKL, "rb") as f:
                history = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load {HISTORY_PKL}: {e}")
    else:
        st.warning(
            "No training history file found. Expected "
            f"`{HISTORY_NPY}` or `{HISTORY_PKL}`."
        )

    if history is not None:
        acc = history.get("accuracy") or history.get("acc")
        val_acc = history.get("val_accuracy") or history.get("val_acc")
        loss = history.get("loss")
        val_loss = history.get("val_loss")

        if acc is not None and val_acc is not None:
            epochs = list(range(1, len(acc) + 1))
            df_acc = pd.DataFrame(
                {
                    "epoch": epochs,
                    "train_accuracy": acc,
                    "val_accuracy": val_acc,
                }
            ).set_index("epoch")

            st.write("### Accuracy")
            st.line_chart(df_acc)

        if loss is not None and val_loss is not None:
            epochs = list(range(1, len(loss) + 1))
            df_loss = pd.DataFrame(
                {
                    "epoch": epochs,
                    "train_loss": loss,
                    "val_loss": val_loss,
                }
            ).set_index("epoch")

            st.write("### Loss")
            st.line_chart(df_loss)


# ---------- TAB 3: Species Classes ----------
with tab_species:
    st.subheader("All Saved Species Classes")

    if label_encoder is None:
        st.error("Label encoder not loaded.")
    else:
        raw_classes = list(label_encoder.classes_)
        pretty_names = [
            c.replace("_", " ").title() for c in raw_classes
        ]
        df_classes = pd.DataFrame(
            {
                "Class Index": list(range(len(pretty_names))),
                "Species": pretty_names,
                "Raw Label": raw_classes,
            }
        )

        st.write(f"Total species: **{len(pretty_names)}**")
        st.dataframe(df_classes, use_container_width=True)
