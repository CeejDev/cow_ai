import cv2
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/raw_videos/cow3.mp4"
ANNOTATION_CSV = "outputs/pose_data/cow3_annotation.csv"
OUTPUT_VIDEO = "outputs/pose_data/cow3_video.mp4"

POSE_LABELS = [
    "nose",
    "poll",
    "withers",
    "spine_1",
    "spine_2",
    "spine_3",
    "hip",
    "tail_base",
    "knee_front_left",
    "knee_front_right",
    "hock_rear_left",
    "hock_rear_right",
    "hoof_front_left",
    "hoof_front_right",
    "hoof_rear_left",
    "hoof_rear_right",
]

SKELETON = [
    ("nose", "poll"),
    ("poll", "withers"),
    ("withers", "spine_1"),
    ("spine_1", "spine_2"),
    ("spine_2", "spine_3"),
    ("spine_3", "hip"),
    ("hip", "tail_base"),

    ("withers", "knee_front_left"),
    ("knee_front_left", "hoof_front_left"),

    ("withers", "knee_front_right"),
    ("knee_front_right", "hoof_front_right"),

    ("hip", "hock_rear_left"),
    ("hock_rear_left", "hoof_rear_left"),

    ("hip", "hock_rear_right"),
    ("hock_rear_right", "hoof_rear_right"),
]

os.makedirs("outputs/pose_data", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(ANNOTATION_CSV)
frames_grouped = df.groupby("frame")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# ============================================================
# CREATE DISPLAY WINDOW (FULL FRAME, NO SCALING)
# ============================================================

window_name = "Manual Gait Annotation Preview"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, width, height)

print("🎬 Rendering manual pose overlay video...")

# ============================================================
# MAIN LOOP
# ============================================================

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in frames_grouped.groups:
        joints = frames_grouped.get_group(frame_idx)

        joint_positions = {}
        for _, row in joints.iterrows():
            x, y = int(row.x), int(row.y)
            joint_positions[row.joint] = (x, y)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        for a, b in SKELETON:
            if a in joint_positions and b in joint_positions:
                cv2.line(
                    frame,
                    joint_positions[a],
                    joint_positions[b],
                    (255, 255, 0),
                    2
                )

    # 🔴 SHOW FULL FRAME WITH ANNOTATIONS
    cv2.imshow(window_name, frame)

    # press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out.write(frame)
    frame_idx += 1

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Manual pose overlay video saved to:\n{OUTPUT_VIDEO}")