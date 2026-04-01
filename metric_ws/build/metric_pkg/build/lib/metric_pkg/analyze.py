#!/usr/bin/env python3

# ============================================================
# JOINT-SPACE BENCHMARK ANALYSIS SCRIPT
# ============================================================
# - Loads 5DOF and 6DOF CSV benchmark files
# - Computes valid research metrics
# - Performs Mann–Whitney U statistical tests
# - Generates publication-ready bar graphs
# - Saves summary tables and figures
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import os

# ============================================================
# 1️⃣ EDIT THESE FILENAMES
# ============================================================

csv_5dof = "benchmark_5dof_joint.csv"
csv_6dof = "benchmark_6dof_joint.csv"

# ============================================================
# 2️⃣ LOAD DATA
# ============================================================

df5 = pd.read_csv(csv_5dof)
df6 = pd.read_csv(csv_6dof)

df5_success = df5[df5["success"] == True]
df6_success = df6[df6["success"] == True]

# ============================================================
# 3️⃣ METRIC COMPUTATION
# ============================================================

def compute_metrics(df, df_success):
    metrics = {}
    metrics["Total Trials"] = len(df)
    metrics["Success Rate (%)"] = df["success"].mean() * 100

    metrics["Average Planning Time (s)"] = df_success["planning_time"].mean()
    metrics["Std Planning Time (s)"] = df_success["planning_time"].std()

    metrics["Average Path Length"] = df_success["path_length"].mean()
    metrics["Std Path Length"] = df_success["path_length"].std()

    metrics["Average Trajectory Duration (s)"] = df_success["trajectory_duration"].mean()
    metrics["Std Trajectory Duration (s)"] = df_success["trajectory_duration"].std()

    metrics["Average Waypoints"] = df_success["num_waypoints"].mean()
    metrics["Std Waypoints"] = df_success["num_waypoints"].std()

    return metrics


metrics_5 = compute_metrics(df5, df5_success)
metrics_6 = compute_metrics(df6, df6_success)

summary_df = pd.DataFrame([metrics_5, metrics_6], index=["5DOF", "6DOF"])
summary_df.to_csv("benchmark_summary_table.csv")

print("\n===== SUMMARY TABLE =====\n")
print(summary_df)

# ============================================================
# 4️⃣ STATISTICAL TESTING (MANN–WHITNEY U)
# ============================================================

def run_stat_test(metric):
    stat, p = mannwhitneyu(
        df5_success[metric].dropna(),
        df6_success[metric].dropna(),
        alternative="two-sided"
    )
    return p

stats_results = {
    "Planning Time p-value": run_stat_test("planning_time"),
    "Path Length p-value": run_stat_test("path_length"),
    "Trajectory Duration p-value": run_stat_test("trajectory_duration"),
    "Waypoints p-value": run_stat_test("num_waypoints"),
}

stats_df = pd.DataFrame([stats_results])
stats_df.to_csv("benchmark_statistical_tests.csv", index=False)

print("\n===== STATISTICAL TEST RESULTS =====\n")
print(stats_df)

# ============================================================
# 5️⃣ BAR GRAPH GENERATION (ONE FIGURE PER METRIC)
# ============================================================

def save_bar_chart(values, labels, title, ylabel, filename):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Success Rate
save_bar_chart(
    [metrics_5["Success Rate (%)"], metrics_6["Success Rate (%)"]],
    ["5DOF", "6DOF"],
    "Planning Success Rate Comparison",
    "Success Rate (%)",
    "success_rate_comparison.png"
)

# Planning Time
save_bar_chart(
    [metrics_5["Average Planning Time (s)"], metrics_6["Average Planning Time (s)"]],
    ["5DOF", "6DOF"],
    "Average Planning Time Comparison",
    "Planning Time (s)",
    "planning_time_comparison.png"
)

# Path Length
save_bar_chart(
    [metrics_5["Average Path Length"], metrics_6["Average Path Length"]],
    ["5DOF", "6DOF"],
    "Average Path Length Comparison",
    "Joint-Space Path Length",
    "path_length_comparison.png"
)

# Trajectory Duration
save_bar_chart(
    [metrics_5["Average Trajectory Duration (s)"], metrics_6["Average Trajectory Duration (s)"]],
    ["5DOF", "6DOF"],
    "Average Trajectory Duration Comparison",
    "Trajectory Duration (s)",
    "trajectory_duration_comparison.png"
)

# Waypoints
save_bar_chart(
    [metrics_5["Average Waypoints"], metrics_6["Average Waypoints"]],
    ["5DOF", "6DOF"],
    "Average Waypoint Count Comparison",
    "Number of Waypoints",
    "waypoints_comparison.png"
)

# ============================================================
# 6️⃣ EFFECT SIZE (PERCENT DIFFERENCE)
# ============================================================

def percent_difference(val5, val6):
    return ((val6 - val5) / val5) * 100

effect_sizes = {
    "Planning Time % Difference (6DOF vs 5DOF)": percent_difference(
        metrics_5["Average Planning Time (s)"],
        metrics_6["Average Planning Time (s)"]
    ),
    "Path Length % Difference": percent_difference(
        metrics_5["Average Path Length"],
        metrics_6["Average Path Length"]
    ),
    "Trajectory Duration % Difference": percent_difference(
        metrics_5["Average Trajectory Duration (s)"],
        metrics_6["Average Trajectory Duration (s)"]
    ),
}

effect_df = pd.DataFrame([effect_sizes])
effect_df.to_csv("benchmark_effect_sizes.csv", index=False)

print("\n===== EFFECT SIZE (% DIFFERENCE) =====\n")
print(effect_df)

print("\nAll analysis files saved successfully.")
