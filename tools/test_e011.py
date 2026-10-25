import numpy as np
import pandas as pd
import subprocess
from possession import spec as spec_module, atd, pipeline
from possession import H, A, D, X
import config

def evaluate_yolo_injection():
    print("1. Loading original tracking data...")
    spec = spec_module.load(None)
    tracking = atd.load(spec, None)
    
    orig_missing = np.isnan(tracking.ball[:, 0]).sum()
    
    print("2. Injecting YOLOv8 recovered ball coordinates...")
    rec = pd.read_csv("recovered_pitch_full.csv")
    frames = rec["frame"].values.astype(int)
    
    # Inject the recovered pitch coordinates into the tracking arrays
    tracking.ball[frames, 0] = rec["pitch_x"].values
    tracking.ball[frames, 1] = rec["pitch_y"].values
    
    new_missing = np.isnan(tracking.ball[:, 0]).sum()
    print(f"-> Restored {orig_missing - new_missing} frames. (Missing dropped from {orig_missing} to {new_missing})")
    
    print("3. Running full HMM pipeline (Pass 1 & Pass 2)...")
    codes = pipeline.label(tracking, spec, config.PARAMS)
    
    print("4. Saving submission CSV...")
    label_map = {H: "H", A: "A", D: "D", X: "X"}
    out_rows = [(f, label_map.get(c, "X")) for f, c in enumerate(codes)]
    
    df = pd.DataFrame(out_rows, columns=["frame", "label"])
    out_path = "evaluation/results/hack_mavericks_e011.csv"
    df.to_csv(out_path, index=False)
    print(f"-> Saved to {out_path}\n")
    
    print("==================================================")
    print("FULL MATCH PLAUSIBILITY STATS (E011)")
    print("==================================================")
    graded = (codes != X)
    h_count = (codes[graded] == H).sum()
    a_count = (codes[graded] == A).sum()
    d_count = (codes[graded] == D).sum()
    total_alive = h_count + a_count
    
    # Calculate possession counts (approximate intervals)
    alive_mask = (codes == H) | (codes == A)
    changes = np.diff(codes[alive_mask]) != 0
    poss_count = changes.sum() + 1
    avg_dur = (total_alive / 25.0) / poss_count if poss_count > 0 else 0
    
    print(f"Possessions:        {poss_count} (Previous baseline: 308)")
    print(f"Average possession: {avg_dur:.1f} s (Previous baseline: 13.3 s)")
    print(f"Dead-ball share:    {d_count / graded.sum():.1%} (Previous baseline: 35.4%)")
    print(f"Home share (alive): {h_count / total_alive:.1%} (Real Barça avg: ~65.0%)")
    print("==================================================\n")
    
    print("5. Running official 5-minute public scorer...")
    subprocess.run(["python", "evaluation/run_evaluation.py"])

if __name__ == "__main__":
    evaluate_yolo_injection()
