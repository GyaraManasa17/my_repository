import json
import numpy as np

def calculate_iso_metrics(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    targets = data['targets']
    achieved = data['achieved']
    
    print("\n" + "="*55)
    print(" 📊 ISO 9283 PERFORMANCE METRICS ".center(55))
    print("="*55 + "\n")
    
    overall_accuracy = []
    overall_repeatability = []

    for pose in["P1", "P2", "P3", "P4", "P5"]:
        target_pos = np.array(targets[pose])
        achieved_poses = np.array(achieved[pose])
        
        if len(achieved_poses) == 0:
            print(f"⚠️ No data recorded for {pose}. Skipping.")
            continue
            
        # 1. Barycenter (Mean of achieved poses)
        barycenter = np.mean(achieved_poses, axis=0)
        
        # 2. Position Accuracy (AP)
        # Distance between commanded target and actual barycenter
        accuracy = np.linalg.norm(barycenter - target_pos) * 1000 # Convert to mm
        
        # 3. Position Repeatability (RP)
        distances_to_barycenter = np.linalg.norm(achieved_poses - barycenter, axis=1)
        l_bar = np.mean(distances_to_barycenter)
        S_l = np.std(distances_to_barycenter, ddof=1)
        repeatability = (l_bar + 3 * S_l) * 1000 # Convert to mm
        
        overall_accuracy.append(accuracy)
        overall_repeatability.append(repeatability)
        
        print(f"--- {pose} ---")
        print(f"  Target:        [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}] m")
        print(f"  Barycenter:    [{barycenter[0]:.4f}, {barycenter[1]:.4f}, {barycenter[2]:.4f}] m")
        print(f"  🎯 Accuracy (AP):     {accuracy:.3f} mm")
        print(f"  🔁 Repeatability (RP): {repeatability:.3f} mm\n")

    if overall_accuracy and overall_repeatability:
        print("="*55)
        print(f" 🏆 AVERAGE WORKSPACE ACCURACY:      {np.mean(overall_accuracy):.3f} mm")
        print(f" 🏆 AVERAGE WORKSPACE REPEATABILITY: {np.mean(overall_repeatability):.3f} mm")
        print("="*55)

if __name__ == "__main__":
    try:
        calculate_iso_metrics('iso_test_results.json')
    except FileNotFoundError:
        print("❌ Error: 'iso_test_results.json' not found. Did you run the execution script yet?")