import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_label_data(db_path, robot_label):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM runs", conn)
        df['Robot_Type'] = robot_label
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading {db_path}: {e}")
        return pd.DataFrame()

print("Loading databases...")
df_4dof = load_and_label_data('4dof.db', '4-DOF Arm')
df_5dof = load_and_label_data('5dof.db', '5-DOF Arm')
df_all = pd.concat([df_4dof, df_5dof], ignore_index=True)

print("Cleaning data...")
# Translate the words 'true' and 'false' into 1 and 0
df_all = df_all.replace({'true': 1, 'false': 0, 'True': 1, 'False': 0})

# ADDED NEW METRICS TO THE CLEANING LIST
metrics_to_convert =[
    'solved', 'time', 'path_plan_length', 'path_plan_smoothness', 
    'final_path_length', 'process_time', 'average_waypoint_distance'
]

for metric in metrics_to_convert:
    if metric in df_all.columns:
        df_all[metric] = pd.to_numeric(df_all[metric], errors='coerce')

df_all['solved'] = df_all['solved'].fillna(0)
df_successful = df_all[df_all['solved'] == 1].copy()

print("Generating 6 graphs...")

# Set up a 2x3 grid (18 units wide by 10 units tall)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
colors =["#FF6B6B", "#4ECDC4"] 
sns.set_palette(sns.color_palette(colors))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Comprehensive Performance: 4-DOF vs 5-DOF Manipulator (300 Poses)", fontsize=20, fontweight='bold')

# --- Plot 1: Success Rate ---
success_rates = df_all.groupby('Robot_Type')['solved'].mean() * 100
success_rates.plot(kind='bar', ax=axes[0, 0], color=colors, edgecolor='black')
axes[0, 0].set_title('Success Rate', fontweight='bold')
axes[0, 0].set_ylabel('Success (%)')
axes[0, 0].set_ylim(0, 105)
axes[0, 0].tick_params(axis='x', rotation=0)

# --- Plot 2: Planning Time ---
sns.boxplot(data=df_successful, x='Robot_Type', y='time', ax=axes[0, 1], width=0.5)
axes[0, 1].set_title('Raw Planning Time', fontweight='bold')
axes[0, 1].set_ylabel('Time (sec)')
axes[0, 1].set_xlabel('')

# --- Plot 3: Raw Path Length ---
sns.boxplot(data=df_successful, x='Robot_Type', y='path_plan_length', ax=axes[0, 2], width=0.5)
axes[0, 2].set_title('Raw Path Length (Before Smoothing)', fontweight='bold')
axes[0, 2].set_ylabel('Length (rad/m)')
axes[0, 2].set_xlabel('')

# --- Plot 4: Final Path Length (Smoothed) ---
if 'final_path_length' in df_successful.columns:
    sns.boxplot(data=df_successful, x='Robot_Type', y='final_path_length', ax=axes[1, 0], width=0.5)
    axes[1, 0].set_title('Final Path Length (After Smoothing)', fontweight='bold')
    axes[1, 0].set_ylabel('Length (rad/m)')
    axes[1, 0].set_xlabel('')

# --- Plot 5: Average Waypoint Distance ---
if 'average_waypoint_distance' in df_successful.columns:
    sns.boxplot(data=df_successful, x='Robot_Type', y='average_waypoint_distance', ax=axes[1, 1], width=0.5)
    axes[1, 1].set_title('Average Waypoint Distance', fontweight='bold')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].set_xlabel('')

# --- Plot 6: Total Process Time ---
if 'process_time' in df_successful.columns:
    sns.boxplot(data=df_successful, x='Robot_Type', y='process_time', ax=axes[1, 2], width=0.5)
    axes[1, 2].set_title('Total Process Time (Plan + Smooth)', fontweight='bold')
    axes[1, 2].set_ylabel('Time (sec)')
    axes[1, 2].set_xlabel('')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("Robot_Comparison_6_Metrics.pdf", dpi=300, bbox_inches='tight')
plt.show()



# import sqlite3
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ==========================================
# # 1. FUNCTION TO LOAD AND PREPARE DATA
# # ==========================================
# def load_and_label_data(db_path, robot_label):
#     """Connects to the SQLite DB, extracts the runs, and adds a label."""
#     try:
#         conn = sqlite3.connect(db_path)
#         # The benchmark metrics are stored in the 'runs' table
#         df = pd.read_sql_query("SELECT * FROM runs", conn)
#         df['Robot_Type'] = robot_label
#         conn.close()
#         return df
#     except Exception as e:
#         print(f"Error loading {db_path}: {e}")
#         return pd.DataFrame()

# # Load both databases
# print("Loading databases...")
# df_4dof = load_and_label_data('4dof.db', '4-DOF Arm')
# df_5dof = load_and_label_data('5dof.db', '5-DOF Arm')

# # Combine them into a single Master DataFrame
# df_all = pd.concat([df_4dof, df_5dof], ignore_index=True)

# # ==========================================
# # 2. DATA CLEANING
# # ==========================================
# # print("Cleaning data...")
# # # Convert numerical columns to actual floats (sometimes SQLite stores them as strings)
# # metrics_to_convert =['solved', 'time', 'path_plan_length', 'path_plan_smoothness']
# # for metric in metrics_to_convert:
# #     if metric in df_all.columns:
# #         df_all[metric] = pd.to_numeric(df_all[metric], errors='coerce')

# # # Fill missing 'solved' values with 0 (Failure)
# # df_all['solved'] = df_all['solved'].fillna(0)

# # # CREATE A DATAFRAME FOR SUCCESSFUL RUNS ONLY
# # # We do this because comparing the "planning time" or "path length" of a failed run 
# # # is scientifically inaccurate for a research paper.
# # df_successful = df_all[df_all['solved'] == 1].copy()

# # ==========================================
# # 2. DATA CLEANING (UPDATED FIX)
# # ==========================================
# print("Cleaning data...")

# # FIX: Translate the words 'true' and 'false' into 1 and 0
# df_all = df_all.replace({'true': 1, 'false': 0, 'True': 1, 'False': 0})

# metrics_to_convert =['solved', 'time', 'path_plan_length', 'path_plan_smoothness']
# for metric in metrics_to_convert:
#     if metric in df_all.columns:
#         df_all[metric] = pd.to_numeric(df_all[metric], errors='coerce')

# df_all['solved'] = df_all['solved'].fillna(0)
# df_successful = df_all[df_all['solved'] == 1].copy()

# # ==========================================
# # 3. PLOTTING PUBLICATION-QUALITY GRAPHS
# # ==========================================
# print("Generating graphs...")

# # Set the visual style to look highly professional (ideal for IEEE/Springer papers)
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
# colors = ["#FF6B6B", "#4ECDC4"] # Distinct, colorblind-friendly colors
# sns.set_palette(sns.color_palette(colors))

# # Create a figure with 4 subplots (2x2 grid)
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# fig.suptitle("Performance Comparison: 4-DOF vs 5-DOF Manipulator (300 Poses)", fontsize=18, fontweight='bold')

# # --- Plot 1: Success Rate (Bar Chart) ---
# # Calculates the percentage of times 'solved' == 1
# success_rates = df_all.groupby('Robot_Type')['solved'].mean() * 100
# success_rates.plot(kind='bar', ax=axes[0, 0], color=colors, edgecolor='black')
# axes[0, 0].set_title('Pose Solving Success Rate', fontweight='bold')
# axes[0, 0].set_ylabel('Success Rate (%)')
# axes[0, 0].set_ylim(0, 105)
# axes[0, 0].tick_params(axis='x', rotation=0)

# # --- Plot 2: Planning Time (Box Plot) ---
# # Shows the distribution of time taken to find a solution (lower is better)
# sns.boxplot(data=df_successful, x='Robot_Type', y='time', ax=axes[0, 1], width=0.5)
# axes[0, 1].set_title('Planning Time (Successful Runs Only)', fontweight='bold')
# axes[0, 1].set_ylabel('Time (seconds)')
# axes[0, 1].set_xlabel('')

# # --- Plot 3: Path Length (Box Plot) ---
# # Shows how efficient the movements are (lower is better)
# sns.boxplot(data=df_successful, x='Robot_Type', y='path_plan_length', ax=axes[1, 0], width=0.5)
# axes[1, 0].set_title('Generated Path Length', fontweight='bold')
# axes[1, 0].set_ylabel('Path Length (radians/meters)')
# axes[1, 0].set_xlabel('')

# # --- Plot 4: Path Smoothness (Box Plot) ---
# # Shows how smooth the trajectory is (lower is generally better, indicating fewer sharp turns)
# if 'path_plan_smoothness' in df_successful.columns:
#     sns.boxplot(data=df_successful, x='Robot_Type', y='path_plan_smoothness', ax=axes[1, 1], width=0.5)
#     axes[1, 1].set_title('Path Smoothness', fontweight='bold')
#     axes[1, 1].set_ylabel('Smoothness Score')
#     axes[1, 1].set_xlabel('')
# else:
#     axes[1, 1].text(0.5, 0.5, 'Smoothness metric not found', ha='center', va='center')

# # Adjust layout to prevent text from overlapping
# plt.tight_layout(rect=[0, 0, 1, 0.96])

# # ==========================================
# # 4. SAVE AND DISPLAY
# # ==========================================
# # Save as a high-resolution PDF and PNG for your LaTeX/Word document
# plt.savefig("Robot_Comparison_Metrics.pdf", dpi=300, bbox_inches='tight')
# plt.savefig("Robot_Comparison_Metrics.png", dpi=300, bbox_inches='tight')
# print("Saved graphs as 'Robot_Comparison_Metrics.pdf' and '.png'")

# # Show the graphs on your screen
# plt.show()

# # ==========================================
# # 5. PRINT STATISTICAL SUMMARY FOR PAPER TEXT
# # ==========================================
# print("\n" + "="*50)
# print("STATISTICAL SUMMARY FOR YOUR RESEARCH PAPER:")
# print("="*50)
# print(f"Total Poses Evaluated: {len(df_4dof)} per robot")
# print(f"\nSuccess Rates:")
# print(f"4-DOF: {success_rates['4-DOF Arm']:.1f}%")
# print(f"5-DOF: {success_rates['5-DOF Arm']:.1f}%")

# print(f"\nAverage Planning Time (Successful Runs):")
# print(f"4-DOF: {df_successful[df_successful['Robot_Type']=='4-DOF Arm']['time'].mean():.4f} seconds")
# print(f"5-DOF: {df_successful[df_successful['Robot_Type']=='5-DOF Arm']['time'].mean():.4f} seconds")

# print(f"\nAverage Path Length (Successful Runs):")
# print(f"4-DOF: {df_successful[df_successful['Robot_Type']=='4-DOF Arm']['path_plan_length'].mean():.4f}")
# print(f"5-DOF: {df_successful[df_successful['Robot_Type']=='5-DOF Arm']['path_plan_length'].mean():.4f}")