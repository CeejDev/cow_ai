import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# CONFIG (LOCKED)
# ============================================================

DATASET_ROOT = "dataset"
SEQUENCE_LENGTH = 60
INPUT_DIM = 32
NUM_CLASSES = 3
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3

LABEL_MAP = {
    "normal": 0,
    "moderate": 1,
    "lame": 2
}

# ============================================================
# DATASET
# ============================================================

class CowPoseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# LOAD DATA
# ============================================================

X, y = [], []

for label_name, label_idx in LABEL_MAP.items():
    folder = os.path.join(DATASET_ROOT, label_name)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            X.append(data)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples")

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

train_ds = CowPoseDataset(X_train, y_train)
val_ds = CowPoseDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ============================================================
# MODEL
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
        # x: (B, T, F)
        x = x.permute(0, 2, 1)        # (B, F, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)        # (B, T, F)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# ============================================================
# TRAINING SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# TRAIN LOOP
# ============================================================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

# ============================================================
# EVALUATION
# ============================================================

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(yb.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=LABEL_MAP.keys()))

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lameness_cnn_lstm.pth")

print("✅ Deep learning model trained and saved")