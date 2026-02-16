import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG (LOCKED)
# ============================================================

ANNOTATION_CSV = "outputs/pose_data/cow1-0_annotation.csv"
OUTPUT_ROOT = "dataset"

SEQUENCE_LENGTH = 60
STRIDE = 10

POSE_LABELS = [
    "nose", "poll", "withers",
    "spine_1", "spine_2", "spine_3",
    "hip", "tail_base",
    "knee_front_left", "knee_front_right",
    "hock_rear_left", "hock_rear_right",
    "hoof_front_left", "hoof_front_right",
    "hoof_rear_left", "hoof_rear_right"
]

# ============================================================
# MANUAL LABEL (CHANGE THIS PER COW)
# ============================================================

COW_LABEL = "lame"   # one of: normal, moderate, lame

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(ANNOTATION_CSV)

frames = sorted(df["frame"].unique())

# Build frame → pose vector
pose_by_frame = {}

for frame in frames:
    frame_data = df[df["frame"] == frame]
    pose_vec = []

    for joint in POSE_LABELS:
        row = frame_data[frame_data["joint"] == joint]
        if row.empty:
            pose_vec.extend([0, 0])
        else:
            pose_vec.extend([row.iloc[0]["x"], row.iloc[0]["y"]])

    pose_by_frame[frame] = np.array(pose_vec, dtype=np.float32)

# ============================================================
# SLICE INTO SEQUENCES
# ============================================================

os.makedirs(os.path.join(OUTPUT_ROOT, COW_LABEL), exist_ok=True)

sequence_id = 0
frame_list = list(pose_by_frame.keys())

for start in range(0, len(frame_list) - SEQUENCE_LENGTH, STRIDE):
    seq_frames = frame_list[start:start + SEQUENCE_LENGTH]

    sequence = np.stack([
        pose_by_frame[f] for f in seq_frames
    ])

    save_path = os.path.join(
        OUTPUT_ROOT,
        COW_LABEL,
        f"cow_seq_{sequence_id}.npy"
    )

    np.save(save_path, sequence)
    sequence_id += 1

print(f"✅ Created {sequence_id} sequences for label '{COW_LABEL}'")