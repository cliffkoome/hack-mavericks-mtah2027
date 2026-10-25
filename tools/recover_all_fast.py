import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO
from possession import atd, features
from possession import spec as spec_module
from starter import load_homography

def recover_all_fast():
    print("Loading tracking and homography data...")
    spec = spec_module.load(None) 
    tracking = atd.load(spec, None)
    H_matrices, err, valid = load_homography.load()

    missing_mask = ~features.ball_visible(tracking)
    graded_mask = np.zeros_like(missing_mask, dtype=bool)
    for first, last in spec.graded_blocks:
        graded_mask[first:last+1] = True
        
    missing_frames = np.flatnonzero(missing_mask & graded_mask)
    missing_set = set(missing_frames) # O(1) lookups
    total_missing = len(missing_frames)
    
    if total_missing == 0:
        print("No missing frames found.")
        return
        
    print(f"Total missing graded frames to recover: {total_missing}")
    print("Loading YOLOv8m model onto GPU...")
    model = YOLO('yolov8m.pt')

    data_dir = os.environ.get("MTAH_DATA_DIR", "data")
    video_path = os.path.join(data_dir, "tactical", "BAR-ATH_tactical.mp4")
    cap = cv2.VideoCapture(video_path)

    # Jump exactly once to the first missing frame
    start_frame = missing_frames[0]
    end_frame = missing_frames[-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    recovered_coords = []
    start_time = time.time()
    frames_processed = 0
    current_frame = start_frame
    
    print("Starting optimized sequential inference...")
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only run heavy YOLO inference if this frame is actually missing the ball
        if current_frame in missing_set and valid[current_frame]:
            # device=0 explicitly forces the computation onto your RTX 4070
            results = model(frame, imgsz=1088, conf=0.05, classes=[32], verbose=False, device=0)
            
            for r in results:
                boxes = r.boxes
                if len(boxes) > 0:
                    sorted_boxes = sorted(boxes, key=lambda x: float(x.conf[0]), reverse=True)
                    box = sorted_boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    conf = float(box.conf[0])
                    
                    if conf > 0.05:
                        H = H_matrices[current_frame]
                        try:
                            H_inv = np.linalg.inv(H)
                            pt = np.array([cx, cy, 1.0])
                            pitch_pt = H_inv @ pt
                            recovered_coords.append({
                                "frame": current_frame,
                                "pitch_x": pitch_pt[0] / pitch_pt[2],
                                "pitch_y": pitch_pt[1] / pitch_pt[2],
                                "conf": conf
                            })
                        except np.linalg.LinAlgError:
                            pass
                            
            frames_processed += 1
            
            # Print progress frequently so you know it isn't stuck
            if frames_processed % 1000 == 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed
                print(f"Recovered {len(recovered_coords)} balls from {frames_processed}/{total_missing} missing frames... ({fps:.1f} inferences/sec)")
        
        current_frame += 1

    cap.release()
    
    out_df = pd.DataFrame(recovered_coords)
    out_df.to_csv("recovered_pitch_full.csv", index=False)
    
    total_time = time.time() - start_time
    print(f"\nFINISHED! Recovered {len(out_df)} balls out of {total_missing} missing frames.")
    print(f"Total time: {total_time/60:.1f} minutes.")

if __name__ == "__main__":
    recover_all_fast()
