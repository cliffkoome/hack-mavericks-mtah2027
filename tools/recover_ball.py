import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from possession import atd, features
from possession import spec as spec_module

def recover_missing_ball():
    print("Loading tracking data...")
    spec = spec_module.load(None) 
    tracking = atd.load(spec, None)

    missing_mask = ~features.ball_visible(tracking)
    # Only look at missing frames that are actually within the graded halves of the match
    graded_mask = np.zeros_like(missing_mask, dtype=bool)
    for first, last in spec.graded_blocks:
        graded_mask[first:last+1] = True
        
    missing_graded_frames = np.flatnonzero(missing_mask & graded_mask)
    print(f"Total missing graded frames to recover: {len(missing_graded_frames)}")

    # UPGRADE 1: Use the Medium model instead of Nano
    print("Loading YOLOv8m model onto GPU...")
    model = YOLO('yolov8m.pt')

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    video_path = os.path.join(data_dir, "tactical", "BAR-ATH_tactical.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"CRITICAL ERROR: Cannot open video at {video_path}")
        return

    recovered = []
    test_frames = missing_graded_frames[:500] 
    print(f"Running high-res inference on {len(test_frames)} frames...")

    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # UPGRADE 2: Process at native ~1080p resolution (1088 is a multiple of 32 for YOLO)
        # UPGRADE 3: Lower the internal confidence threshold to 0.05 to allow faint detections
        results = model(frame, imgsz=1088, conf=0.05, classes=[32], verbose=False)
        
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                # Sort boxes by confidence if multiple are found
                sorted_boxes = sorted(boxes, key=lambda x: float(x.conf[0]), reverse=True)
                box = sorted_boxes[0]
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                conf = float(box.conf[0])
                
                if conf > 0.05: 
                    recovered.append({
                        "frame": frame_idx, 
                        "pixel_x": cx, 
                        "pixel_y": cy, 
                        "conf": conf
                    })

    cap.release()

    if recovered:
        df = pd.DataFrame(recovered)
        df.to_csv("recovered_pixels.csv", index=False)
        print(f"Success! Recovered {len(df)} ball positions out of {len(test_frames)} tested.")
    else:
        print("Still no balls found in the test batch.")

if __name__ == "__main__":
    recover_missing_ball()
