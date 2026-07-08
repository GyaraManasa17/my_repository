#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
# UPDATE THESE TWO LINES to your exact generated CSV filenames!
FILE_5DOF = "omx_all_reachable_poses_5dof.csv"
FILE_6DOF = "omx_all_reachable_poses_6dof.csv"

# Global plotting settings for Publication Quality
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.autolayout': True,
    'figure.dpi': 300,            # High resolution for research papers
    'savefig.bbox': 'tight'
})

# Colors
COLOR_5DOF = "#1f77b4" # Deep Blue
COLOR_6DOF = "#ff7f0e" # Rich Orange

def load_and_prep_data():
    print("Loading data...")
    df5 = pd.read_csv(FILE_5DOF)
    df6 = pd.read_csv(FILE_6DOF)
    
    # Add labels
    df5['Config'] = '5-DOF (Original)'
    df6['Config'] = '6-DOF (Enhanced)'
    
    # Combine
    df = pd.concat([df5, df6], ignore_index=True)
    
    # Calculate advanced metrics
    # Radial reach: Distance from base (0,0,0)
    df['Radial_Reach'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    # Floor footprint reach: Distance from base on the XY plane
    df['XY_Reach'] = np.sqrt(df['x']**2 + df['y']**2)
    
    # Convert Angles to Degrees for easier reading by non-experts
    df['roll_deg'] = np.degrees(df['roll'])
    df['pitch_deg'] = np.degrees(df['pitch'])
    df['yaw_deg'] = np.degrees(df['yaw'])
    
    return df, df5, df6

def plot_master_dashboard(df):
    print("Generating Master Dashboard...")
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Kinematic Workspace Comparison: 5-DOF vs 6-DOF", fontsize=18, fontweight='bold')
    
    # Top-Left: Radial Reach Density (Violin Plot)
    sns.violinplot(data=df, x='Config', y='Radial_Reach', palette=[COLOR_5DOF, COLOR_6DOF], ax=axs[0, 0])
    axs[0, 0].set_title("Total Reach Capabilities")
    axs[0, 0].set_ylabel("Radial Distance from Base [meters]")
    axs[0, 0].set_xlabel("")
    
    # Top-Right: Vertical (Z-axis) Distribution
    sns.kdeplot(data=df, x='z', hue='Config', fill=True, palette=[COLOR_5DOF, COLOR_6DOF], alpha=0.5, ax=axs[0, 1])
    axs[0, 1].set_title("Vertical Working Height (Z-Axis)")
    axs[0, 1].set_xlabel("Height from base [meters]")
    axs[0, 1].set_ylabel("Pose Density")
    
    # Bottom-Left: Footprint (XY) Distribution
    sns.kdeplot(data=df, x='XY_Reach', hue='Config', fill=True, palette=[COLOR_5DOF, COLOR_6DOF], alpha=0.5, ax=axs[1, 0])
    axs[1, 0].set_title("Tabletop Footprint Spread (XY Plane)")
    axs[1, 0].set_xlabel("Horizontal Distance from base [meters]")
    axs[1, 0].set_ylabel("Pose Density")
    
    # Bottom-Right: Tool Orientation Flexibility (Dexterity Proxy)
    # We use Standard Deviation of angles to show "flexibility"
    std_data = df.groupby('Config')[['roll_deg', 'pitch_deg', 'yaw_deg']].std().reset_index()
    std_data_melted = std_data.melt(id_vars='Config', var_name='Axis', value_name='Variance (Degrees)')
    sns.barplot(data=std_data_melted, x='Axis', y='Variance (Degrees)', hue='Config', palette=[COLOR_5DOF, COLOR_6DOF], ax=axs[1, 1])
    axs[1, 1].set_title("Orientation Flexibility (Higher = More Dexterous)")
    axs[1, 1].set_xticklabels(['Roll', 'Pitch', 'Yaw'])
    axs[1, 1].set_xlabel("Tool Rotation Axis")
    
    plt.savefig("Fig1_Master_Dashboard.png")
    plt.close()

def plot_topdown_heatmaps(df5, df6):
    print("Generating XY Heatmaps...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle("Top-Down Reachability Footprint (XY Plane)", fontsize=16, fontweight='bold')
    
    sns.kdeplot(x=df5['x'], y=df5['y'], cmap="Blues", fill=True, thresh=0.05, ax=axs[0])
    axs[0].set_title("5-DOF Workspace Density")
    axs[0].set_xlabel("X Coordinate (Front/Back) [m]")
    axs[0].set_ylabel("Y Coordinate (Left/Right) [m]")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    sns.kdeplot(x=df6['x'], y=df6['y'], cmap="Oranges", fill=True, thresh=0.05, ax=axs[1])
    axs[1].set_title("6-DOF Workspace Density")
    axs[1].set_xlabel("X Coordinate (Front/Back) [m]")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    # Ensure axes are equal so the physical circles aren't warped
    axs[0].set_aspect('equal', 'box')
    axs[1].set_aspect('equal', 'box')
    
    plt.savefig("Fig2_TopDown_Heatmaps.png")
    plt.close()

def plot_orientation_dexterity(df):
    print("Generating Orientation Distributions...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("End-Effector Orientation Capabilities", fontsize=16, fontweight='bold')
    
    axes_list =['roll_deg', 'pitch_deg', 'yaw_deg']
    titles = ["Roll Angle Distribution", "Pitch Angle Distribution", "Yaw Angle Distribution"]
    
    for i in range(3):
        sns.kdeplot(data=df, x=axes_list[i], hue='Config', fill=True, palette=[COLOR_5DOF, COLOR_6DOF], alpha=0.4, ax=axs[i])
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Angle [Degrees]")
        axs[i].set_ylabel("Frequency")
        axs[i].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("Fig3_Orientation_Dexterity.png")
    plt.close()

def plot_3d_overlay(df5, df6):
    print("Generating 3D Scatter Overlay...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample points so the plot is readable (2000 points each)
    sample_5 = df5.sample(n=min(2000, len(df5)), random_state=42)
    sample_6 = df6.sample(n=min(2000, len(df6)), random_state=42)
    
    # Plot 6-DOF first (larger, in orange)
    ax.scatter(sample_6['x'], sample_6['y'], sample_6['z'], 
               c=COLOR_6DOF, s=5, alpha=0.3, label='6-DOF Reach')
    
    # Plot 5-DOF on top (smaller, in blue)
    ax.scatter(sample_5['x'], sample_5['y'], sample_5['z'], 
               c=COLOR_5DOF, s=5, alpha=0.7, label='5-DOF Reach')
    
    ax.set_title("3D Workspace Envelope Overlay", fontsize=16, fontweight='bold')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Height (Z) [m]')
    
    # Set equal limits to maintain aspect ratio
    max_range = 0.5 # 0.5 meters
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-0.25, 0.5])
    
    ax.legend(markerscale=3)
    
    # We save an image, but if you run this in a Jupyter Notebook, it's interactive!
    plt.savefig("Fig4_3D_Overlay.png")
    plt.close()

def main():
    try:
        df, df5, df6 = load_and_prep_data()
    except FileNotFoundError:
        print("\n❌ Error: Could not find the CSV files.")
        print("Please edit FILE_5DOF and FILE_6DOF at the top of this script to match your CSV filenames.")
        return

    plot_master_dashboard(df)
    plot_topdown_heatmaps(df5, df6)
    plot_orientation_dexterity(df)
    plot_3d_overlay(df5, df6)
    
    print("\n✅ Success! All publication-ready figures have been generated and saved in your folder:")
    print("   1. Fig1_Master_Dashboard.png")
    print("   2. Fig2_TopDown_Heatmaps.png")
    print("   3. Fig3_Orientation_Dexterity.png")
    print("   4. Fig4_3D_Overlay.png")

if __name__ == "__main__":
    main()