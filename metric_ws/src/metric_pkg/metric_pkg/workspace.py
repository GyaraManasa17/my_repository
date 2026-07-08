#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

def plot_workspace(csv_filename=None):
    # If no file is provided, auto-find the newest one
    if csv_filename is None:
        files = glob.glob("omx_workspace_boundary_*.csv")
        if not files:
            print("❌ No CSV files found in the current directory.")
            return
        # Get the most recently created file
        csv_filename = max(files, key=os.path.getctime)
        
    print(f"📂 Loading data from: {csv_filename}")
    
    # Read the data
    df = pd.read_csv(csv_filename)
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot, colored by Z-height (makes it look 3D and readable)
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=10, alpha=0.8, edgecolors='none')

    # Plot the robot's base origin (0,0,0) as a red star
    ax.scatter([0], [0], [0], color='red', marker='*', s=200, label="Robot Base (0,0,0)")

    # Add titles and labels
    ax.set_title("OpenManipulator-X Reachable Workspace Boundary", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (meters)", fontweight='bold')
    ax.set_ylabel("Y (meters)", fontweight='bold')
    ax.set_zlabel("Z (meters)", fontweight='bold')
    
    # Add a color bar mapping to the Z height
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Z Height (meters)')
    
    ax.legend()

    # FORCE EQUAL ASPECT RATIO (Crucial for robotics so the bubble isn't stretched)
    # Matplotlib 3D needs this specific hack to look geometrically correct
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print("✨ Rendering plot... (Close the window to exit)")
    plt.show()

if __name__ == "__main__":
    # You can pass a specific filename via terminal: python3 plot_workspace.py my_file.csv
    if len(sys.argv) > 1:
        plot_workspace(sys.argv[1])
    else:
        plot_workspace()