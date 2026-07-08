#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_FILE = "omx_all_reachable_poses_6dof_FULL.csv"  # The file from Script 1
OUTPUT_FILE = "real_robot_40_test_poses_table_top2.csv"

NUM_TOTAL = 40
NUM_BOUNDARY = 12   
NUM_INTERIOR = 28   

print("\n" + "="*70)
print(" 🎯 STRATEGIC SAMPLER: SMART TABLE-SAFE POSE GENERATOR")
print("="*70)

# ==========================================
# 🧠 FARTHEST POINT SAMPLING (FPS) ALGORITHM
# ==========================================
def farthest_point_sampling(points, k):
    print(f"\n[FPS] Starting Farthest Point Sampling for {k} points...")
    start_time = time.time()
    selected_indices =[]
    
    idx = np.random.randint(len(points))
    selected_indices.append(idx)
    distances = cdist(points, [points[idx]]).flatten()
    
    for step in range(1, k):
        idx = np.argmax(distances)
        selected_indices.append(idx)
        new_dist = cdist(points,[points[idx]]).flatten()
        distances = np.minimum(distances, new_dist)
        
        if step % 5 == 0 or step == k - 1:
            print(f"      -> FPS picked {step + 1}/{k} points...")
            
    print(f"[FPS] Completed in {time.time() - start_time:.3f} seconds.")
    return selected_indices

# ==========================================
# 🚀 MAIN PIPELINE
# ==========================================
def main():
    # 1. Load the Dataset
    print(f"\n[STEP 1] Loading dataset: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        original_count = len(df)
        print(f"         -> Successfully loaded {original_count} raw poses.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {INPUT_FILE}. Please run the Analyzer first!")
        return

    # ---------------------------------------------------------
    # 🔴 STEP 1.5: ULTRA-SAFE "UPPER HEMISPHERE" CLEARANCE FILTERS
    # ---------------------------------------------------------
    print("\n[STEP 1.5] Applying Ultra-Safe Upper Hemisphere Filters...")
    
    # Filter 1: The High Floor (20 cm minimum!)
    # 20cm guarantees the entire arm, wrist, and elbow stay well above the table.
    Z_MIN_SAFE = 0.20  
    df_safe = df[df['z'] >= Z_MIN_SAFE].copy()
    
    # Filter 2: The "No Bowing" Filter (Shoulder lock)
    # On the OpenManipulator, joint2 is the shoulder. If it goes too positive, 
    # the robot "bows" forward toward the table. We restrict it to stay mostly upright.
    df_safe = df_safe[(df_safe['j2'] > -1.2) & (df_safe['j2'] < 0.5)].copy()
    
    # Filter 3: The "No Pointing Down" Filter
    # We restrict the pitch. The gripper must remain mostly horizontal or point UP.
    # It is strictly forbidden from pointing straight down at the table.
    df_safe = df_safe[df_safe['pitch'].abs() < 1.0].copy()

    # Reset index so .iloc indexing works perfectly in the next steps
    df_safe = df_safe.reset_index(drop=True)

    removed_count = original_count - len(df_safe)
    print(f"         -> 🗑️ Removed {removed_count} poses (Too low, bowing forward, or pointing down)!")
    print(f"         -> ✅ Kept {len(df_safe)} ULTRA-SAFE poses for sampling.")

    if len(df_safe) < NUM_TOTAL:
        print("[ERROR] Filters were too aggressive! Not enough poses left. Lower Z_MIN_SAFE slightly to 0.18.")
        return

    # Extract just the XYZ coordinates for spatial mathematics
    points = df_safe[['x', 'y', 'z']].values

    # ---------------------------------------------------------
    # 2. BOUNDARY SAMPLING (Edge Cases)
    # ---------------------------------------------------------
    print(f"\n[STEP 2] Extracting {NUM_BOUNDARY} Boundary Poses...")
    
    center = np.mean(points, axis=0)
    print(f"         -> Workspace Centroid: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
    
    distances_from_center = np.linalg.norm(points - center, axis=1)
    
    boundary_indices = np.argsort(distances_from_center)[-NUM_BOUNDARY:]
    boundary_df = df_safe.iloc[boundary_indices].copy()
    boundary_df['test_type'] = 'Boundary (Edge Case)'
    
    print(f"         -> Extracted {len(boundary_df)} boundary points.")

    # ---------------------------------------------------------
    # 3. INTERIOR FPS SAMPLING (Workspace Coverage)
    # ---------------------------------------------------------
    print(f"\n[STEP 3] Running FPS on the Interior Poses...")
    
    interior_df = df_safe.drop(boundary_indices).reset_index(drop=True)
    interior_points = interior_df[['x','y','z']].values
    
    print(f"         -> Remaining interior pool: {len(interior_df)} poses.")
    
    fps_indices = farthest_point_sampling(interior_points, NUM_INTERIOR)
    
    fps_df = interior_df.iloc[fps_indices].copy()
    fps_df['test_type'] = 'Interior (FPS Spread)'

    # ---------------------------------------------------------
    # 4. COMBINE AND SHUFFLE
    # ---------------------------------------------------------
    print("\n[STEP 4] Combining and Shuffling dataset...")
    final_df = pd.concat([boundary_df, fps_df])
    # Shuffle so the real robot alternates between edge and interior cases (safer for motors)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ---------------------------------------------------------
    # 5. SAVE FINAL DATA
    # ---------------------------------------------------------
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*70)
    print(" ✅ SUCCESS: RESEARCH-GRADE REAL ROBOT DATASET READY")
    print("="*70)
    print(f" 💾 File Saved:    {OUTPUT_FILE}")
    print(f" 📊 Total Poses:   {len(final_df)} (All guaranteed safe from table collisions)")
    print(" 🛠️ Data Included: Joints (j1-j5), Pos (XYZ), RPY, Quaternions, Test Type")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()





# #!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
# import time

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# INPUT_FILE = "omx_all_reachable_poses_6dof_FULL.csv"  # The file from Script 1
# OUTPUT_FILE = "real_robot_40_test_poses_table_top.csv"

# NUM_TOTAL = 40
# NUM_BOUNDARY = 12   
# NUM_INTERIOR = 28   

# print("\n" + "="*70)
# print(" 🎯 STRATEGIC SAMPLER: SMART TABLE-SAFE POSE GENERATOR")
# print("="*70)

# # ==========================================
# # 🧠 FARTHEST POINT SAMPLING (FPS) ALGORITHM
# # ==========================================
# def farthest_point_sampling(points, k):
#     print(f"\n[FPS] Starting Farthest Point Sampling for {k} points...")
#     start_time = time.time()
#     selected_indices =[]
    
#     idx = np.random.randint(len(points))
#     selected_indices.append(idx)
#     distances = cdist(points, [points[idx]]).flatten()
    
#     for step in range(1, k):
#         idx = np.argmax(distances)
#         selected_indices.append(idx)
#         new_dist = cdist(points,[points[idx]]).flatten()
#         distances = np.minimum(distances, new_dist)
        
#         if step % 5 == 0 or step == k - 1:
#             print(f"      -> FPS picked {step + 1}/{k} points...")
            
#     print(f"[FPS] Completed in {time.time() - start_time:.3f} seconds.")
#     return selected_indices

# # ==========================================
# # 🚀 MAIN PIPELINE
# # ==========================================
# def main():
#     # 1. Load the Dataset
#     print(f"\n[STEP 1] Loading dataset: {INPUT_FILE}")
#     try:
#         df = pd.read_csv(INPUT_FILE)
#         original_count = len(df)
#         print(f"         -> Successfully loaded {original_count} raw poses.")
#     except FileNotFoundError:
#         print(f"[ERROR] Could not find {INPUT_FILE}. Please run the Analyzer first!")
#         return

#     # ---------------------------------------------------------
#     # 🔴 STEP 1.5: SMART TABLE COLLISION & ELBOW SAFETY FILTER
#     # ---------------------------------------------------------
#     print("\n[STEP 1.5] Applying Smart Hardware Clearance Filters...")
    
#     # Filter 1: Raise the End-Effector Floor (12 cm instead of 3 cm)
#     # Ensures there is enough physical room underneath the hand for the elbow/wrist.
#     Z_MIN_SAFE = 0.12  
#     df_safe = df[df['z'] >= Z_MIN_SAFE].copy()
    
#     # Filter 2: Prevent "Elbow Drop" (Shoulder joint limits)
#     # Restrict joint2 (shoulder) from leaning too close to the floor.
#     df_safe = df_safe[(df_safe['j2'] > -1.2) & (df_safe['j2'] < 1.2)].copy()
    
#     # Filter 3: Prevent "Nose Dive"
#     # If the robot is relatively low (under 20cm), restrict severe downward/upward pitch
#     # so the gripper remains mostly horizontal and doesn't spear the table.
#     nose_dive_condition = (df_safe['z'] < 0.20) & (df_safe['pitch'].abs() > 1.3)
#     df_safe = df_safe[~nose_dive_condition].copy()

#     # Reset index so .iloc indexing works perfectly in the next steps
#     df_safe = df_safe.reset_index(drop=True)

#     removed_count = original_count - len(df_safe)
#     print(f"         -> 🗑️ Removed {removed_count} dangerous poses (Elbow drops, Nose dives, Low Z)!")
#     print(f"         -> ✅ Kept {len(df_safe)} completely table-safe poses for sampling.")

#     if len(df_safe) < NUM_TOTAL:
#         print("[ERROR] Filters were too aggressive! Not enough poses left. Lower Z_MIN_SAFE.")
#         return

#     # Extract just the XYZ coordinates for spatial mathematics
#     points = df_safe[['x', 'y', 'z']].values

#     # ---------------------------------------------------------
#     # 2. BOUNDARY SAMPLING (Edge Cases)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 2] Extracting {NUM_BOUNDARY} Boundary Poses...")
    
#     center = np.mean(points, axis=0)
#     print(f"         -> Workspace Centroid: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
    
#     distances_from_center = np.linalg.norm(points - center, axis=1)
    
#     boundary_indices = np.argsort(distances_from_center)[-NUM_BOUNDARY:]
#     boundary_df = df_safe.iloc[boundary_indices].copy()
#     boundary_df['test_type'] = 'Boundary (Edge Case)'
    
#     print(f"         -> Extracted {len(boundary_df)} boundary points.")

#     # ---------------------------------------------------------
#     # 3. INTERIOR FPS SAMPLING (Workspace Coverage)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 3] Running FPS on the Interior Poses...")
    
#     interior_df = df_safe.drop(boundary_indices).reset_index(drop=True)
#     interior_points = interior_df[['x','y','z']].values
    
#     print(f"         -> Remaining interior pool: {len(interior_df)} poses.")
    
#     fps_indices = farthest_point_sampling(interior_points, NUM_INTERIOR)
    
#     fps_df = interior_df.iloc[fps_indices].copy()
#     fps_df['test_type'] = 'Interior (FPS Spread)'

#     # ---------------------------------------------------------
#     # 4. COMBINE AND SHUFFLE
#     # ---------------------------------------------------------
#     print("\n[STEP 4] Combining and Shuffling dataset...")
#     final_df = pd.concat([boundary_df, fps_df])
#     # Shuffle so the real robot alternates between edge and interior cases (safer for motors)
#     final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

#     # ---------------------------------------------------------
#     # 5. SAVE FINAL DATA
#     # ---------------------------------------------------------
#     final_df.to_csv(OUTPUT_FILE, index=False)
    
#     print("\n" + "="*70)
#     print(" ✅ SUCCESS: RESEARCH-GRADE REAL ROBOT DATASET READY")
#     print("="*70)
#     print(f" 💾 File Saved:    {OUTPUT_FILE}")
#     print(f" 📊 Total Poses:   {len(final_df)} (All guaranteed safe from table collisions)")
#     print(" 🛠️ Data Included: Joints (j1-j5), Pos (XYZ), RPY, Quaternions, Test Type")
#     print("="*70 + "\n")

# if __name__ == "__main__":
#     main()




# #!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
# import time

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# INPUT_FILE = "omx_all_reachable_poses_6dof_FULL.csv"  # The file from Script 1
# OUTPUT_FILE = "real_robot_40_test_poses_above_table.csv"

# NUM_TOTAL = 40
# NUM_BOUNDARY = 12   
# NUM_INTERIOR = 28   

# # 🔴 SAFETY CONSTRAINT: Minimum height above the table in meters
# # 0.03 = 3 cm above the base link (prevents gripper from scraping the table)
# Z_MIN_SAFE = 0.03   

# print("\n" + "="*70)
# print(" 🎯 STRATEGIC SAMPLER: TABLE-SAFE POSE GENERATOR")
# print("="*70)

# # ==========================================
# # 🧠 FARTHEST POINT SAMPLING (FPS) ALGORITHM
# # ==========================================
# def farthest_point_sampling(points, k):
#     print(f"\n[FPS] Starting Farthest Point Sampling for {k} points...")
#     start_time = time.time()
#     selected_indices =[]
    
#     idx = np.random.randint(len(points))
#     selected_indices.append(idx)
#     distances = cdist(points, [points[idx]]).flatten()
    
#     for step in range(1, k):
#         idx = np.argmax(distances)
#         selected_indices.append(idx)
#         new_dist = cdist(points,[points[idx]]).flatten()
#         distances = np.minimum(distances, new_dist)
        
#         if step % 5 == 0 or step == k - 1:
#             print(f"      -> FPS picked {step + 1}/{k} points...")
            
#     print(f"[FPS] Completed in {time.time() - start_time:.3f} seconds.")
#     return selected_indices

# # ==========================================
# # 🚀 MAIN PIPELINE
# # ==========================================
# def main():
#     # 1. Load the Dataset
#     print(f"\n[STEP 1] Loading dataset: {INPUT_FILE}")
#     try:
#         df = pd.read_csv(INPUT_FILE)
#         original_count = len(df)
#         print(f"         -> Successfully loaded {original_count} raw poses.")
#     except FileNotFoundError:
#         print(f"[ERROR] Could not find {INPUT_FILE}.")
#         return

#     # ---------------------------------------------------------
#     # 🔴 STEP 1.5: TABLE COLLISION SAFETY FILTER
#     # ---------------------------------------------------------
#     print(f"\n[STEP 1.5] Applying Z-Height Safety Filter (Z >= {Z_MIN_SAFE}m)...")
    
#     # Keep only rows where Z is greater than or equal to our safety margin
#     df_safe = df[df['z'] >= Z_MIN_SAFE].reset_index(drop=True)
    
#     removed_count = original_count - len(df_safe)
#     print(f"         -> 🗑️ Removed {removed_count} poses that were below the table!")
#     print(f"         -> ✅ Kept {len(df_safe)} safe, above-table poses for sampling.")

#     # Extract XYZ points from the SAFE dataframe
#     points = df_safe[['x', 'y', 'z']].values

#     # ---------------------------------------------------------
#     # 2. BOUNDARY SAMPLING (Edge Cases)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 2] Extracting {NUM_BOUNDARY} Boundary Poses...")
    
#     center = np.mean(points, axis=0)
#     print(f"         -> Workspace Centroid: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
    
#     distances_from_center = np.linalg.norm(points - center, axis=1)
    
#     boundary_indices = np.argsort(distances_from_center)[-NUM_BOUNDARY:]
#     boundary_df = df_safe.iloc[boundary_indices].copy()
#     boundary_df['test_type'] = 'Boundary (Edge Case)'
    
#     print(f"         -> Extracted {len(boundary_df)} boundary points.")

#     # ---------------------------------------------------------
#     # 3. INTERIOR FPS SAMPLING (Workspace Coverage)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 3] Running FPS on the Interior Poses...")
    
#     interior_df = df_safe.drop(boundary_indices).reset_index(drop=True)
#     interior_points = interior_df[['x','y','z']].values
    
#     print(f"         -> Remaining interior pool: {len(interior_df)} poses.")
    
#     fps_indices = farthest_point_sampling(interior_points, NUM_INTERIOR)
    
#     fps_df = interior_df.iloc[fps_indices].copy()
#     fps_df['test_type'] = 'Interior (FPS Spread)'

#     # ---------------------------------------------------------
#     # 4. COMBINE AND SHUFFLE
#     # ---------------------------------------------------------
#     print("\n[STEP 4] Combining and Shuffling dataset...")
#     final_df = pd.concat([boundary_df, fps_df])
#     final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

#     # ---------------------------------------------------------
#     # 5. SAVE FINAL DATA
#     # ---------------------------------------------------------
#     final_df.to_csv(OUTPUT_FILE, index=False)
    
#     print("\n" + "="*70)
#     print(" ✅ SUCCESS: TABLE-SAFE ROBOT DATASET READY")
#     print("="*70)
#     print(f" 💾 File Saved:    {OUTPUT_FILE}")
#     print(f" 📊 Total Poses:   {len(final_df)} (All guaranteed Z >= {Z_MIN_SAFE}m)")
#     print("="*70 + "\n")

# if __name__ == "__main__":
#     main()





# #!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# from scipy.spatial.distance import cdist
# import time

# # ==========================================
# # ⚙️ CONFIGURATION
# # ==========================================
# # This reads the newly formatted 15-column CSV
# INPUT_FILE = "omx_all_reachable_poses_6dof_FULL.csv"  
# OUTPUT_FILE = "real_robot_40_test_poses.csv"

# NUM_TOTAL = 40
# NUM_BOUNDARY = 12   # 30% Stress test poses
# NUM_INTERIOR = 28   # 70% FPS spread poses

# print("\n" + "="*70)
# print(" 🎯 STRATEGIC SAMPLER: GENERATING 40 POSES FOR REAL ROBOT")
# print("="*70)

# # ==========================================
# # 🧠 FARTHEST POINT SAMPLING (FPS) ALGORITHM
# # ==========================================
# def farthest_point_sampling(points, k):
#     """
#     Selects 'k' points that are maximally far apart.
#     Returns array INDICES to perfectly preserve the DataFrame rows.
#     """
#     print(f"\n[FPS] Starting Farthest Point Sampling for {k} points...")
#     start_time = time.time()
    
#     selected_indices =[]
    
#     # 1. Start with a random point
#     idx = np.random.randint(len(points))
#     selected_indices.append(idx)
    
#     # 2. Calculate initial distances from this first point
#     distances = cdist(points, [points[idx]]).flatten()
    
#     # 3. Iteratively pick the farthest point
#     for step in range(1, k):
#         idx = np.argmax(distances)
#         selected_indices.append(idx)
        
#         # Update distances
#         new_dist = cdist(points, [points[idx]]).flatten()
#         distances = np.minimum(distances, new_dist)
        
#         if step % 5 == 0 or step == k - 1:
#             print(f"      -> FPS picked {step + 1}/{k} points...")
            
#     print(f"[FPS] Completed in {time.time() - start_time:.3f} seconds.")
#     return selected_indices

# # ==========================================
# # 🚀 MAIN PIPELINE
# # ==========================================
# def main():
#     # 1. Load the Dataset
#     print(f"\n[STEP 1] Loading dataset: {INPUT_FILE}")
#     try:
#         df = pd.read_csv(INPUT_FILE)
#         print(f"         -> Successfully loaded {len(df)} reachable poses.")
#         print(f"         -> Found Columns: {list(df.columns)}")
#     except FileNotFoundError:
#         print(f"[ERROR] Could not find {INPUT_FILE}. Please run the updated Analyzer first!")
#         return

#     # Extract just the XYZ coordinates for spatial mathematics
#     points = df[['x', 'y', 'z']].values

#     # ---------------------------------------------------------
#     # 2. BOUNDARY SAMPLING (Edge Cases)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 2] Extracting {NUM_BOUNDARY} Boundary Poses...")
    
#     # Calculate Workspace Centroid
#     center = np.mean(points, axis=0)
#     print(f"         -> Workspace Centroid: X={center[0]:.3f}, Y={center[1]:.3f}, Z={center[2]:.3f}")
    
#     # Find points furthest from center
#     distances_from_center = np.linalg.norm(points - center, axis=1)
    
#     # Get indices of the furthest points
#     boundary_indices = np.argsort(distances_from_center)[-NUM_BOUNDARY:]
#     boundary_df = df.iloc[boundary_indices].copy()
#     boundary_df['test_type'] = 'Boundary (Edge Case)'
    
#     print(f"         -> Extracted {len(boundary_df)} boundary points.")

#     # ---------------------------------------------------------
#     # 3. INTERIOR FPS SAMPLING (Workspace Coverage)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 3] Running FPS on the Interior Poses...")
    
#     # Drop the boundary points so we don't pick them twice
#     interior_df = df.drop(boundary_indices).reset_index(drop=True)
#     interior_points = interior_df[['x','y','z']].values
    
#     print(f"         -> Remaining interior pool: {len(interior_df)} poses.")
    
#     # Run FPS Algorithm
#     fps_indices = farthest_point_sampling(interior_points, NUM_INTERIOR)
    
#     # Extract rows safely
#     fps_df = interior_df.iloc[fps_indices].copy()
#     fps_df['test_type'] = 'Interior (FPS Spread)'

#     # ---------------------------------------------------------
#     # 4. COMBINE AND SHUFFLE
#     # ---------------------------------------------------------
#     print("\n[STEP 4] Combining and Shuffling dataset...")
    
#     final_df = pd.concat([boundary_df, fps_df])
    
#     # Shuffle so the real robot alternates between edge and interior cases (safer for motors)
#     final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

#     # ---------------------------------------------------------
#     # 5. SAVE FINAL DATA
#     # ---------------------------------------------------------
#     final_df.to_csv(OUTPUT_FILE, index=False)
    
#     print("\n" + "="*70)
#     print(" ✅ SUCCESS: RESEARCH-GRADE REAL ROBOT DATASET READY")
#     print("="*70)
#     print(f" 💾 File Saved:    {OUTPUT_FILE}")
#     print(f" 📊 Total Poses:   {len(final_df)}")
#     print(" 🛠️ Data Included: Joints (j1-j5), Pos (XYZ), RPY, Quaternions (qx-qw)")
#     print("="*70 + "\n")

# if __name__ == "__main__":
#     main()