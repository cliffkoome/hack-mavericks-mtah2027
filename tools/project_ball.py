import pandas as pd
import numpy as np
from starter import load_homography

def project_to_pitch():
    print("Loading recovered pixels...")
    try:
        df = pd.read_csv("recovered_pixels.csv")
    except FileNotFoundError:
        print("ERROR: recovered_pixels.csv not found.")
        return
        
    print("Loading homography matrices...")
    # The starter code provides a load() function that returns the matrices
    H_matrices, err, valid = load_homography.load()
    
    pitch_coords = []
    for _, row in df.iterrows():
        frame = int(row['frame'])
        px, py = row['pixel_x'], row['pixel_y']
        
        # Only project if we have a valid camera matrix for this frame
        if valid[frame]:
            H = H_matrices[frame]
            
            # Calculate the inverse matrix to go from Pixels -> Pitch
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                continue
            
            # Apply the inverse projection
            pt = np.array([px, py, 1.0])
            pitch_pt = H_inv @ pt
            
            # Normalize by the Z component
            pitch_x = pitch_pt[0] / pitch_pt[2]
            pitch_y = pitch_pt[1] / pitch_pt[2]
            
            pitch_coords.append({
                "frame": frame,
                "pitch_x": pitch_x,
                "pitch_y": pitch_y,
                "conf": row['conf']
            })
    
    out_df = pd.DataFrame(pitch_coords)
    out_df.to_csv("recovered_pitch.csv", index=False)
    print(f"Successfully projected {len(out_df)} frames to pitch coordinates.")
    
    # Check if the coordinates look like valid normalized pitch coordinates (0.0 to 1.0)
    print("\nSample of projected coordinates:")
    print(out_df[['frame', 'pitch_x', 'pitch_y']].head())

if __name__ == "__main__":
    project_to_pitch()
