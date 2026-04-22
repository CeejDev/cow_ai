import cv2
import os
from ultralytics import YOLO

# ============================================================
# 1. PATH SETUP
# ============================================================
video_path = "data/raw_videos/vid2.mp4"
model_path = "yolov8n.pt"

tracked_video_dir = "outputs/tracked_videos"
cow_tracks_dir = "data/tracked_cows"
output_video_name = "cow_tracker.mp4"

os.makedirs(tracked_video_dir, exist_ok=True)
os.makedirs(cow_tracks_dir, exist_ok=True)

# ============================================================
# 2. LOAD YOLO MODEL
# ============================================================
model = YOLO(model_path)

# ============================================================
# 3. VIDEO CAPTURE
# ============================================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Could not open video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_skip = int(fps / 2) if fps > 0 else 15

# ============================================================
# 4. VIDEO WRITER
# ============================================================
output_video_path = os.path.join(tracked_video_dir, output_video_name)
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ============================================================
# 5. CREATE DISPLAY WINDOW
# ============================================================
window_name = "Cow Detection & Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, width, height)

# ============================================================
# 6. TRACKING LOOP
# ============================================================
frame_id = 0
# Define specific pixel offsets for more control
TOP_OFFSET = 60   # Increase this to move the box higher
LEFT_OFFSET = 60  # Increase this to move the box further left
BOTTOM_OFFSET = 20
RIGHT_OFFSET = 20

print("🚀 Processing started...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[19], conf=0.5, verbose=False)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id

        for box, track_id in zip(boxes, ids):
            x1_raw, y1_raw, x2_raw, y2_raw = map(int, box)
            cow_id = int(track_id)

            # --- APPLY PADDING TO CREATE THE EXPANDED BOX ---
            x1 = max(0, x1_raw - LEFT_OFFSET)
            y1 = max(0, y1_raw - TOP_OFFSET)
            x2 = min(width, x2_raw + RIGHT_OFFSET)
            y2 = min(height, y2_raw + BOTTOM_OFFSET)

            # Draw the EXPANDED box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Cow {cow_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Only save the cropped image 2 times per second
            if frame_id % frame_skip == 0:
                cow_folder = os.path.join(cow_tracks_dir, f"cow_{cow_id}")
                os.makedirs(cow_folder, exist_ok=True)

                cow_crop = frame[y1:y2, x1:x2]

                if cow_crop.size > 0:
                    cv2.imwrite(os.path.join(cow_folder, f"frame_{frame_id}.jpg"), cow_crop)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out.write(frame)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processing complete")