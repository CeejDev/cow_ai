import streamlit as st
import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import cv2
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/lameness_cnn_lstm.pth"
DATASET_ROOT = "dataset"
SAVED_COWS_DIR = "outputs/saved_cows"
TEMP_VIDEO_PATH = "outputs/temp_uploaded_video.mp4"

LABEL_MAP = {
    0: "Normal",
    1: "Moderate",
    2: "Lame"
}

INPUT_DIM = 32
NUM_CLASSES = 3

os.makedirs(SAVED_COWS_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ============================================================
# SESSION STATE (NON-WIDGET ONLY)
# ============================================================

if "final_pred" not in st.session_state:
    st.session_state.final_pred = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None

# ============================================================
# MODEL DEFINITION
# ============================================================

class CNN_LSTM_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(INPUT_DIM, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(layout="wide")

st.title("🐄 AI-Based Cow Lameness Detection (Demo)")
st.markdown("Upload a cow walking video and receive a lameness score.")

uploaded_video = st.file_uploader(
    "Upload cow walking video",
    type=["mp4", "avi"]
)

cow_type = st.selectbox(
    "Select cow gait pattern for demo inference",
    ["normal", "moderate", "lame"]
)

# ============================================================
# RUN INFERENCE
# ============================================================

if uploaded_video and st.button("Run Lameness Analysis"):
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(uploaded_video.read())

    st.video(TEMP_VIDEO_PATH)

    predictions = []
    seq_folder = os.path.join(DATASET_ROOT, cow_type)

    for file in os.listdir(seq_folder):
        if not file.endswith(".npy"):
            continue

        seq = np.load(os.path.join(seq_folder, file))
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(seq)
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)

    final_pred = Counter(predictions).most_common(1)[0][0]

    st.session_state.final_pred = final_pred
    st.session_state.video_path = TEMP_VIDEO_PATH

    st.success(
        f"Final Lameness Score: **{LABEL_MAP[final_pred]} ({final_pred + 1})**"
    )

# ============================================================
# SAVE RESULT (SAFE RESET)
# ============================================================

if st.session_state.final_pred is not None:
    if st.button("💾 Save Cow Result"):
        cap = cv2.VideoCapture(st.session_state.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()

        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label = LABEL_MAP[st.session_state.final_pred]
            img_name = f"cow_{timestamp}_{label}.jpg"
            img_path = os.path.join(SAVED_COWS_DIR, img_name)

            cv2.imwrite(img_path, frame)

            st.success("✅ Cow and lameness score saved!")

            # Reset ONLY analysis state (合法做法)
            st.session_state.final_pred = None
            st.session_state.video_path = None

            st.rerun()

# ============================================================
# SIDEBAR DASHBOARD (WITH DELETE)
# ============================================================

st.sidebar.title("📊 Saved Cows & Lameness Scores")

saved_files = sorted(os.listdir(SAVED_COWS_DIR), reverse=True)

if not saved_files:
    st.sidebar.info("No saved cows yet.")
else:
    for file in saved_files:
        img_path = os.path.join(SAVED_COWS_DIR, file)
        label = file.split("_")[-1].replace(".jpg", "")

        st.sidebar.image(
            img_path,
            caption=f"Lameness: {label}",
            use_container_width=True
        )

        # 🗑 Delete button
        if st.sidebar.button(f"Delete {file}", key=f"del_{file}"):
            os.remove(img_path)
            st.rerun()