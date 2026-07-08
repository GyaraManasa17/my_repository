import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import os

def select_iso_9283_poses(csv_file_path):
    print(f"Loading dataset: {csv_file_path}")
    # 1. Load your generated 10,000 points
    df = pd.read_csv(csv_file_path)
    
    # -------------------------------------------------------------
    # 🛑 NEW: FILTER OUT UNDERGROUND POSES
    # -------------------------------------------------------------
    # Keep only poses where Z is at least 0.05 meters (5 cm) above the floor
    safe_z_threshold = 0.05
    initial_count = len(df)
    df = df[df['z'] >= safe_z_threshold]
    filtered_count = len(df)
    
    print(f"⚠️ Filtered out {initial_count - filtered_count} underground poses.")
    print(f"✅ Remaining safe above-ground poses: {filtered_count}")
    
    # If we accidentally filtered everything out, stop the script
    if filtered_count < 5:
        print("❌ Error: Not enough poses left after filtering!")
        return None

    # Extract just the X, Y, Z coordinates from the filtered dataset
    positions = df[['x', 'y', 'z']].values

    # 2. Find the bounding box of your safe, above-ground workspace
    min_bounds = np.min(positions, axis=0)
    max_bounds = np.max(positions, axis=0)
    
    # 3. Define the ISO Test Cube
    center = (max_bounds + min_bounds) / 2.0
    ranges = max_bounds - min_bounds
    
    # Use 80% of the smallest range as the cube side-length (L) to ensure it's safely inside the workspace
    L = np.min(ranges) * 0.8 
    half_L = L / 2.0

    # 4. Calculate the 5 Ideal ISO 9283 Points
    offset = L * 0.10 
    dist = half_L - offset

    ideal_p1 = center
    ideal_p2 = center + np.array([dist, dist, dist])
    ideal_p3 = center + np.array([-dist, -dist, dist])
    ideal_p4 = center + np.array([-dist, dist, -dist])
    ideal_p5 = center + np.array([dist, -dist, -dist])

    ideal_points =[ideal_p1, ideal_p2, ideal_p3, ideal_p4, ideal_p5]
    point_names =["P1 (Center)", "P2 (Top-Right-Front)", "P3 (Top-Left-Back)", "P4 (Bottom-Left-Front)", "P5 (Bottom-Right-Back)"]

    # 5. Find the closest ACTUAL reachable poses in your dataset using a KD-Tree
    tree = KDTree(positions)
    
    selected_indices =[]
    print("\n--- ISO 9283 Selected Poses ---")
    
    for i, ideal_pt in enumerate(ideal_points):
        # Query the KD-Tree for the 1 nearest neighbor
        distance, index = tree.query(ideal_pt, k=1)
        selected_indices.append(index)
        
        actual_pt = positions[index]
        print(f"\n{point_names[i]}:")
        print(f"  Ideal target:[{ideal_pt[0]:.4f}, {ideal_pt[1]:.4f}, {ideal_pt[2]:.4f}]")
        print(f"  Actual in CSV:[{actual_pt[0]:.4f}, {actual_pt[1]:.4f}, {actual_pt[2]:.4f}]")
        print(f"  Distance err:  {distance*1000:.2f} mm")

    # 6. Extract the full joint configurations for these 5 points
    selected_poses_df = df.iloc[selected_indices]
    
    # Save these 5 specific poses to a new CSV so you can use them in your hardware tests
    output_file = "iso_9283_test_poses.csv"
    selected_poses_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved the 5 ISO standard safe test configurations to: {output_file}")
    
    return selected_poses_df

if __name__ == "__main__":
    # Ensure this points to your correct dataset
    LATEST_CSV = "fk_metrics_6/workspace_fk_dataset_20260323_082903.csv" 
    
    if os.path.exists(LATEST_CSV):
        select_iso_9283_poses(LATEST_CSV)
    else:
        print(f"❌ Could not find {LATEST_CSV}. Please update the script with your actual filename.")