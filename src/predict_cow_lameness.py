import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/lameness_cnn_lstm.pth"
DATASET_ROOT = "dataset"

LABEL_MAP = {
    0: "normal",
    1: "moderate",
    2: "lame"
}

INPUT_DIM = 32
NUM_CLASSES = 3

# ============================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
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
        x = x.permute(0, 2, 1)   # (B, F, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)   # (B, T, F)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM_Model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("✅ Model loaded")

# ============================================================
# PREDICT ONE COW (MULTI-SEQUENCE)
# ============================================================

def predict_cow_from_folder(sequence_folder):
    predictions = []

    for file in os.listdir(sequence_folder):
        if not file.endswith(".npy"):
            continue

        seq = np.load(os.path.join(sequence_folder, file))
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(seq)
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)

    if not predictions:
        raise ValueError("No sequences found for this cow.")

    vote = Counter(predictions).most_common(1)[0][0]
    return vote, predictions

# ============================================================
# DEMO: RUN ON ONE COW CLASS FOLDER
# ============================================================

if __name__ == "__main__":
    # CHANGE THIS to test different cows
    cow_folder = "dataset/normal"

    cow_pred, all_preds = predict_cow_from_folder(cow_folder)

    print("\nSequence-level predictions:")
    for p in all_preds:
        print(LABEL_MAP[p])

    print("\nFINAL COW-LEVEL LAMENESS SCORE:")
    print(f"{LABEL_MAP[cow_pred].upper()} ({cow_pred + 1})")
