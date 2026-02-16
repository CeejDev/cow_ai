import cv2
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "data/raw_videos/cow3.mp4"
OUTPUT_CSV = "outputs/pose_data/sample_annotation.csv"

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800

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

os.makedirs("outputs/pose_data", exist_ok=True)

# ============================================================
# VIDEO SETUP
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame")

orig_h, orig_w = frame.shape[:2]

scale = min(
    MAX_DISPLAY_WIDTH / orig_w,
    MAX_DISPLAY_HEIGHT / orig_h,
    1.0
)

disp_w = int(orig_w * scale)
disp_h = int(orig_h * scale)

scale_x = orig_w / disp_w
scale_y = orig_h / disp_h

# ============================================================
# STATE
# ============================================================

frame_idx = 0
annotations = []          # all frames
current_points = []       # current frame only

# ============================================================
# DRAW FUNCTION
# ============================================================

def redraw(frame_img):
    img = cv2.resize(frame_img, (disp_w, disp_h))

    for label, x, y in current_points:
        dx = int(x / scale_x)
        dy = int(y / scale_y)

        cv2.circle(img, (dx, dy), 4, (0, 0, 255), -1)
        cv2.putText(
            img, label,
            (dx + 4, dy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

    cv2.imshow("Manual Annotation", img)

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_points) < len(POSE_LABELS):
            ox = int(x * scale_x)
            oy = int(y * scale_y)
            label = POSE_LABELS[len(current_points)]
            current_points.append((label, ox, oy))
            print(f"✔ Frame {frame_idx} | {label}: ({ox}, {oy})")
            redraw(frame)

# ============================================================
# UI SETUP
# ============================================================

cv2.namedWindow("Manual Annotation", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Manual Annotation", mouse_callback)

print("\n🖱️ CONTROLS")
print("Left Click  → Add point")
print("Backspace  → Undo last point")
print("R          → Redo current frame")
print("B          → Go back one frame")
print("N          → Save frame & next")
print("ESC        → Exit\n")

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    redraw(frame)
    key = cv2.waitKey(0) & 0xFF

    # ESC → exit
    if key == 27:
        break

    # Undo
    if key == 8 and current_points:
        removed = current_points.pop()
        print(f"↩ Removed {removed[0]}")
        redraw(frame)

    # Redo frame
    if key in (ord("r"), ord("R")):
        print(f"🔄 Redo frame {frame_idx}")
        current_points.clear()
        redraw(frame)

    # Save + next frame
    if key in (ord("n"), ord("N")):
        if len(current_points) != len(POSE_LABELS):
            print("⚠️ Not all points placed yet")
            continue

        for label, x, y in current_points:
            annotations.append({
                "frame": frame_idx,
                "joint": label,
                "x": x,
                "y": y
            })

        print(f"✅ Saved frame {frame_idx}")

        current_points.clear()
        frame_idx += 1

        ret, frame = cap.read()
        if not ret:
            print("🎬 End of video")
            break

    # Go back one frame
    if key in (ord("b"), ord("B")):
        if frame_idx == 0:
            continue

        frame_idx -= 1
        annotations = [a for a in annotations if a["frame"] != frame_idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        current_points.clear()

        print(f"⏪ Back to frame {frame_idx}")
        redraw(frame)

cap.release()
cv2.destroyAllWindows()

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(annotations)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n📄 Manual annotations saved to: {OUTPUT_CSV}")