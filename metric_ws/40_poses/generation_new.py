#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
INPUT_FILE = "omx_safe_base_poses.csv"  
OUTPUT_FILE = "real_robot_40_test_poses_table_top.csv"
EXECUTION_ORDER_FILE = "execution_joint_commands.csv" 

NUM_TOTAL = 40
NUM_BOUNDARY = 12   
NUM_INTERIOR = 28   

print("\n" + "="*70)
print(" 🎯 STRATEGIC SAMPLER: RESEARCH-GRADE POSE GENERATOR")
print("="*70)

def farthest_point_sampling(points, k):
    print(f"\n[FPS] Starting Normalized 6-DOF Farthest Point Sampling for {k} points...")
    start_time = time.time()
    selected_indices = []
    
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

def main():
    print(f"\n[STEP 1] Loading dataset: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        original_count = len(df)
        print(f"         -> Successfully loaded {original_count} raw poses.")
    except FileNotFoundError:
        print(f"[ERROR] Could not find {INPUT_FILE}.")
        return

    # ---------------------------------------------------------
    # 🔴 STEP 1.5: ULTRA-SAFE & SMOOTHNESS FILTERS
    # ---------------------------------------------------------
    print("\n[STEP 1.5] Applying Ultra-Safe Upper Hemisphere Filters...")
    
    # ✅ FIX 1: Increased Z margin to 0.25 for absolute hardware safety
    Z_MIN_SAFE = 0.25  
    df_safe = df[df['z'] >= Z_MIN_SAFE].copy()
    
    df_safe = df_safe[(df_safe['j2'] > -1.2) & (df_safe['j2'] < 0.5)].copy()
    df_safe = df_safe[df_safe['pitch'].abs() < 1.0].copy()
    
    # ✅ FIX 2: Clamp Wrist Rotation to prevent cable snapping / wrapping
    df_safe = df_safe[df_safe['j5'].abs() < 2.5].copy()

    JOINT_DELTA_MAX = 1.5
    df_safe = df_safe.sort_values(by=['j1']).reset_index(drop=True)
    smooth_mask = (df_safe[['j1','j2','j3','j4','j5']].diff().fillna(0).abs() < JOINT_DELTA_MAX).all(axis=1)
    df_safe = df_safe[smooth_mask].reset_index(drop=True)

    removed_count = original_count - len(df_safe)
    print(f"         -> 🗑️ Removed {removed_count} unsafe/jerky/wrapped poses.")
    print(f"         -> ✅ Kept {len(df_safe)} ULTRA-SAFE, smooth poses for sampling.")

    if len(df_safe) < NUM_TOTAL:
        print("[ERROR] Filters were too aggressive! Not enough poses left.")
        return

    # ---------------------------------------------------------
    # 2. BOUNDARY SAMPLING (Absolute Extremes)
    # ---------------------------------------------------------
    print(f"\n[STEP 2] Extracting {NUM_BOUNDARY} Absolute Boundary Poses...")
    
    extreme_indices = [
        df_safe['x'].idxmax(), df_safe['x'].idxmin(),
        df_safe['y'].idxmax(), df_safe['y'].idxmin(),
        df_safe['z'].idxmax(), df_safe['z'].idxmin()
    ]
    extreme_indices = list(set(extreme_indices))
    boundary_df = df_safe.loc[extreme_indices].copy()
    
    rem_needed = NUM_BOUNDARY - len(boundary_df)
    if rem_needed > 0:
        points_xyz = df_safe[['x', 'y', 'z']].values
        center = np.mean(points_xyz, axis=0)
        distances = np.linalg.norm(points_xyz - center, axis=1)
        sorted_indices = np.argsort(distances)[::-1] 
        
        added = 0
        for idx in sorted_indices:
            if idx not in extreme_indices:
                boundary_df = pd.concat([boundary_df, df_safe.iloc[[idx]]])
                extreme_indices.append(idx)
                added += 1
                if added == rem_needed: break

    boundary_df['test_type'] = 'Boundary (Absolute Extreme)'
    print(f"         -> Extracted {len(boundary_df)} true boundary points.")

    # ---------------------------------------------------------
    # 3. INTERIOR FPS SAMPLING (Workspace Coverage + Normalized Orientation)
    # ---------------------------------------------------------
    print(f"\n[STEP 3] Running Normalized FPS on the Interior Poses...")
    
    interior_df = df_safe.drop(extreme_indices).reset_index(drop=True)
    
    pos = interior_df[['x', 'y', 'z']].values
    ori = interior_df[['roll', 'pitch', 'yaw']].values
    
    pos_std = pos.std(axis=0)
    pos_std[pos_std == 0] = 1e-6  
    pos_norm = (pos - pos.mean(axis=0)) / pos_std
    
    ori_norm = ori / np.pi  
    
    interior_points_6d_normalized = np.hstack([pos_norm, ori_norm])
    
    fps_indices = farthest_point_sampling(interior_points_6d_normalized, NUM_INTERIOR)
    
    fps_df = interior_df.iloc[fps_indices].copy()
    fps_df['test_type'] = 'Interior (6-DOF Spread)'

    # ---------------------------------------------------------
    # 4. COMBINE, VALIDATE & SAVE
    # ---------------------------------------------------------
    print("\n[STEP 4] Validating and saving dataset...")
    final_df = pd.concat([boundary_df, fps_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    assert len(final_df) == NUM_TOTAL, f"Mismatch in final pose count! Got {len(final_df)}"
    assert final_df['z'].min() >= 0.24, "Unsafe Z detected slipping through filters!"

    final_df['sampling_method'] = 'Absolute Boundaries + Normalized 6DOF FPS'
    final_df['safety_filters'] = 'Z>=0.25, j2<0.5, pitch<1.0, Smoothness<1.5, |j5|<2.5'

    final_df.to_csv(OUTPUT_FILE, index=False)
    
    execution_df = final_df[['j1','j2','j3','j4','j5']]
    execution_df.to_csv(EXECUTION_ORDER_FILE, index=False)
    
    print("\n" + "="*70)
    print(" ✅ SUCCESS: RESEARCH-GRADE REAL ROBOT DATASET READY")
    print("="*70)
    print(f" 💾 Main Dataset Saved:    {OUTPUT_FILE}")
    print(f" 💾 Execution Order Saved: {EXECUTION_ORDER_FILE}")
    print(f" 📊 Total Poses:           {len(final_df)}")
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
# INPUT_FILE = "omx_safe_base_poses.csv"  
# OUTPUT_FILE = "real_robot_40_test_poses_table_top.csv"
# EXECUTION_ORDER_FILE = "execution_joint_commands.csv" 

# NUM_TOTAL = 40
# NUM_BOUNDARY = 12   
# NUM_INTERIOR = 28   

# print("\n" + "="*70)
# print(" 🎯 STRATEGIC SAMPLER: RESEARCH-GRADE POSE GENERATOR")
# print("="*70)

# def farthest_point_sampling(points, k):
#     print(f"\n[FPS] Starting Normalized 6-DOF Farthest Point Sampling for {k} points...")
#     start_time = time.time()
#     selected_indices = []
    
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

# def main():
#     print(f"\n[STEP 1] Loading dataset: {INPUT_FILE}")
#     try:
#         df = pd.read_csv(INPUT_FILE)
#         original_count = len(df)
#         print(f"         -> Successfully loaded {original_count} raw poses.")
#     except FileNotFoundError:
#         print(f"[ERROR] Could not find {INPUT_FILE}.")
#         return

#     # ---------------------------------------------------------
#     # 🔴 STEP 1.5: ULTRA-SAFE & SMOOTHNESS FILTERS
#     # ---------------------------------------------------------
#     print("\n[STEP 1.5] Applying Ultra-Safe Upper Hemisphere Filters...")
    
#     Z_MIN_SAFE = 0.20  
#     df_safe = df[df['z'] >= Z_MIN_SAFE].copy()
#     df_safe = df_safe[(df_safe['j2'] > -1.2) & (df_safe['j2'] < 0.5)].copy()
#     df_safe = df_safe[df_safe['pitch'].abs() < 1.0].copy()

#     JOINT_DELTA_MAX = 1.5
#     df_safe = df_safe.sort_values(by=['j1']).reset_index(drop=True)
#     smooth_mask = (df_safe[['j1','j2','j3','j4','j5']].diff().fillna(0).abs() < JOINT_DELTA_MAX).all(axis=1)
#     df_safe = df_safe[smooth_mask].reset_index(drop=True)

#     removed_count = original_count - len(df_safe)
#     print(f"         -> 🗑️ Removed {removed_count} unsafe/jerky poses.")
#     print(f"         -> ✅ Kept {len(df_safe)} ULTRA-SAFE, smooth poses for sampling.")

#     if len(df_safe) < NUM_TOTAL:
#         print("[ERROR] Filters were too aggressive! Not enough poses left.")
#         return

#     # ---------------------------------------------------------
#     # 2. BOUNDARY SAMPLING (Absolute Extremes)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 2] Extracting {NUM_BOUNDARY} Absolute Boundary Poses...")
    
#     extreme_indices = [
#         df_safe['x'].idxmax(), df_safe['x'].idxmin(),
#         df_safe['y'].idxmax(), df_safe['y'].idxmin(),
#         df_safe['z'].idxmax(), df_safe['z'].idxmin()
#     ]
#     extreme_indices = list(set(extreme_indices))
#     boundary_df = df_safe.loc[extreme_indices].copy()
    
#     rem_needed = NUM_BOUNDARY - len(boundary_df)
#     if rem_needed > 0:
#         points_xyz = df_safe[['x', 'y', 'z']].values
#         center = np.mean(points_xyz, axis=0)
#         distances = np.linalg.norm(points_xyz - center, axis=1)
#         sorted_indices = np.argsort(distances)[::-1] 
        
#         added = 0
#         for idx in sorted_indices:
#             if idx not in extreme_indices:
#                 boundary_df = pd.concat([boundary_df, df_safe.iloc[[idx]]])
#                 extreme_indices.append(idx)
#                 added += 1
#                 if added == rem_needed: break

#     boundary_df['test_type'] = 'Boundary (Absolute Extreme)'
#     print(f"         -> Extracted {len(boundary_df)} true boundary points.")

#     # ---------------------------------------------------------
#     # 3. INTERIOR FPS SAMPLING (Workspace Coverage + Normalized Orientation)
#     # ---------------------------------------------------------
#     print(f"\n[STEP 3] Running Normalized FPS on the Interior Poses...")
    
#     interior_df = df_safe.drop(extreme_indices).reset_index(drop=True)
    
#     # ✅ Normalized FPS Feature Scaling (Position vs Angles)
#     pos = interior_df[['x', 'y', 'z']].values
#     ori = interior_df[['roll', 'pitch', 'yaw']].values
    
#     # Z-score normalization for position
#     pos_std = pos.std(axis=0)
#     pos_std[pos_std == 0] = 1e-6  
#     pos_norm = (pos - pos.mean(axis=0)) / pos_std
    
#     # Min-Max normalization for orientation (Radians to [-1, 1])
#     ori_norm = ori / np.pi  
    
#     interior_points_6d_normalized = np.hstack([pos_norm, ori_norm])
    
#     fps_indices = farthest_point_sampling(interior_points_6d_normalized, NUM_INTERIOR)
    
#     fps_df = interior_df.iloc[fps_indices].copy()
#     fps_df['test_type'] = 'Interior (6-DOF Spread)'

#     # ---------------------------------------------------------
#     # 4. COMBINE, VALIDATE & SAVE
#     # ---------------------------------------------------------
#     print("\n[STEP 4] Validating and saving dataset...")
#     final_df = pd.concat([boundary_df, fps_df]).sample(frac=1, random_state=42).reset_index(drop=True)

#     assert len(final_df) == NUM_TOTAL, f"Mismatch in final pose count! Got {len(final_df)}"
#     assert final_df['z'].min() >= 0.18, "Unsafe Z detected slipping through filters!"

#     final_df['sampling_method'] = 'Absolute Boundaries + Normalized 6DOF FPS'
#     final_df['safety_filters'] = 'Z>=0.20, j2<0.5, pitch<1.0, Smoothness<1.5'

#     final_df.to_csv(OUTPUT_FILE, index=False)
    
#     execution_df = final_df[['j1','j2','j3','j4','j5']]
#     execution_df.to_csv(EXECUTION_ORDER_FILE, index=False)
    
#     print("\n" + "="*70)
#     print(" ✅ SUCCESS: RESEARCH-GRADE REAL ROBOT DATASET READY")
#     print("="*70)
#     print(f" 💾 Main Dataset Saved:    {OUTPUT_FILE}")
#     print(f" 💾 Execution Order Saved: {EXECUTION_ORDER_FILE}")
#     print(f" 📊 Total Poses:           {len(final_df)}")
#     print("="*70 + "\n")

# if __name__ == "__main__":
#     main()