"""
IEEE-LEVEL Planning Time Statistical Analysis
============================================

Comparative benchmarking for allowed planning times:
5s, 10s, 15s

Includes:
- Mean ± 95% CI
- ANOVA + Effect Size (η²)
- Linear regression + R²
- Publication-quality plots (600 DPI)
- Statistical interpretation report

Author: Research-grade automated pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

# ==============================
# USER INPUT
# ==============================

csv_files = {
    5: "200trials_6dof_5.csv",
    10: "200trials_6dof_end_10.csv",
    15: "200trials_6dof_15.csv"
}

output_dir = "ieee_planning_time_results"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

data = {}
for allowed_time, file in csv_files.items():
    df = pd.read_csv(file)
    df["allowed_time"] = allowed_time
    data[allowed_time] = df

combined_df = pd.concat(data.values(), ignore_index=True)

# ==============================
# IEEE PLOT STYLE
# ==============================

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "figure.figsize": (8, 6),
    "legend.frameon": True
})

# ==============================
# SUMMARY STATISTICS
# ==============================

summary_rows = []

metrics = [
    "planning_time",
    "wall_time",
    "path_length_joint",
    "trajectory_duration",
    "elapsed_time_sec"
]

for allowed_time, df in data.items():

    row = {}
    row["allowed_time"] = allowed_time
    row["n_trials"] = len(df)
    row["success_rate_%"] = df["success"].mean() * 100

    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        ci_low, ci_high = stats.t.interval(
            0.95,
            len(df[metric]) - 1,
            loc=mean,
            scale=stats.sem(df[metric])
        )

        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_CI_low"] = ci_low
        row[f"{metric}_CI_high"] = ci_high

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values("allowed_time")
summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)

# ==============================
# ANOVA + EFFECT SIZE
# ==============================

anova_results = []
report_lines = []

for metric in ["success"] + metrics:

    groups = [data[t][metric] for t in sorted(data.keys())]
    F, p = stats.f_oneway(*groups)

    grand_mean = combined_df[metric].mean()
    ss_between = sum(
        len(data[t][metric]) * (data[t][metric].mean() - grand_mean) ** 2
        for t in data
    )
    ss_total = sum((combined_df[metric] - grand_mean) ** 2)
    eta_sq = ss_between / ss_total

    anova_results.append({
        "metric": metric,
        "F_value": F,
        "p_value": p,
        "eta_squared": eta_sq
    })

    significance = "statistically significant" if p < 0.05 else "not statistically significant"

    report_lines.append(
        f"{metric}: F={F:.3f}, p={p:.4f}, eta²={eta_sq:.3f} → {significance}"
    )

anova_df = pd.DataFrame(anova_results)
anova_df.to_csv(os.path.join(output_dir, "anova_results.csv"), index=False)

# ==============================
# REGRESSION ANALYSIS
# ==============================

regression_results = []

for metric in ["success"] + metrics:

    slope, intercept, r, p_val, std_err = stats.linregress(
        combined_df["allowed_time"],
        combined_df[metric]
    )

    regression_results.append({
        "metric": metric,
        "slope": slope,
        "R_squared": r ** 2,
        "p_value": p_val
    })

reg_df = pd.DataFrame(regression_results)
reg_df.to_csv(os.path.join(output_dir, "regression_results.csv"), index=False)

# ==============================
# INFORMATIVE IEEE PLOTS
# ==============================

def ieee_plot(metric, ylabel, filename):

    plt.figure()

    x = summary_df["allowed_time"]
    means = summary_df[f"{metric}_mean"]
    ci_low = summary_df[f"{metric}_CI_low"]
    ci_high = summary_df[f"{metric}_CI_high"]

    plt.plot(x, means, marker='o', label="Mean")
    plt.fill_between(x, ci_low, ci_high, alpha=0.2, label="95% CI")

    # Regression
    slope, intercept, r, p_val, _ = stats.linregress(
        combined_df["allowed_time"],
        combined_df[metric]
    )
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, linestyle="--",
             label=f"Trend (R²={r**2:.3f})")

    # ANOVA result
    anova_row = anova_df[anova_df["metric"] == metric].iloc[0]
    plt.title(
        f"{ylabel} vs Allowed Planning Time\n"
        f"ANOVA p={anova_row['p_value']:.4f}, η²={anova_row['eta_squared']:.3f}"
    )

    plt.xlabel("Allowed Planning Time (seconds)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=600)
    plt.close()

# Generate metric plots
ieee_plot("planning_time", "Planning Time Used (s)", "planning_time_ieee.png")
ieee_plot("path_length_joint", "Path Length (Joint Space)", "path_length_ieee.png")
ieee_plot("trajectory_duration", "Trajectory Duration (s)", "trajectory_duration_ieee.png")
ieee_plot("wall_time", "Wall Time (s)", "wall_time_ieee.png")

# Success rate plot
plt.figure()
x = summary_df["allowed_time"]
y = summary_df["success_rate_%"]

plt.plot(x, y, marker='o')
plt.title("Planning Success Rate vs Allowed Time")
plt.xlabel("Allowed Planning Time (seconds)")
plt.ylabel("Success Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "success_rate_ieee.png"), dpi=600)
plt.close()

# Boxplot (distribution insight)
plt.figure()
combined_df.boxplot(column="planning_time", by="allowed_time")
plt.title("Planning Time Distribution by Allowed Time")
plt.suptitle("")
plt.xlabel("Allowed Planning Time (seconds)")
plt.ylabel("Planning Time Used (s)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "planning_time_boxplot_ieee.png"), dpi=600)
plt.close()

# ==============================
# SAVE STATISTICAL REPORT
# ==============================

with open(os.path.join(output_dir, "statistical_report.txt"), "w") as f:
    f.write("IEEE Statistical Analysis Report\n")
    f.write("================================\n\n")
    for line in report_lines:
        f.write(line + "\n")

print("\n✅ IEEE-level analysis complete.")
print(f"Results saved in folder: {output_dir}")




# """
# Planning Time Comparative Statistical Analysis
# ==============================================

# Research-grade benchmarking comparison for:
# Allowed planning time = 5, 10, 15, 20 seconds

# Author: Auto-generated statistical analysis pipeline
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from scipy import stats
# from sklearn.linear_model import LinearRegression

# # ==============================
# # 🔹 USER INPUT SECTION
# # ==============================

# csv_files = {
#     5: "200trials_6dof_5.csv",
#     10: "200trials_6dof_end_10.csv",
#     15: "200trials_6dof_15.csv"
#     # 20: "file_20s.csv"
# }

# output_dir = "planning_time_analysis_results"
# os.makedirs(output_dir, exist_ok=True)

# # ==============================
# # 🔹 LOAD DATA
# # ==============================

# data = {}

# for allowed_time, file in csv_files.items():
#     df = pd.read_csv(file)
#     df["allowed_time"] = allowed_time
#     data[allowed_time] = df

# combined_df = pd.concat(data.values(), ignore_index=True)

# # ==============================
# # 🔹 SUMMARY STATISTICS
# # ==============================

# summary_rows = []

# for allowed_time, df in data.items():

#     success_rate = df["success"].mean() * 100
#     failure_rate = 100 - success_rate

#     metrics = [
#         "planning_time",
#         "wall_time",
#         "path_length_joint",
#         "trajectory_duration",
#         "elapsed_time_sec"
#     ]

#     row = {
#         "allowed_time": allowed_time,
#         "success_rate_%": success_rate,
#         "failure_rate_%": failure_rate,
#     }

#     for metric in metrics:
#         mean = df[metric].mean()
#         std = df[metric].std()
#         ci_low, ci_high = stats.t.interval(
#             0.95,
#             len(df[metric]) - 1,
#             loc=mean,
#             scale=stats.sem(df[metric])
#         )

#         row[f"{metric}_mean"] = mean
#         row[f"{metric}_std"] = std
#         row[f"{metric}_CI_low"] = ci_low
#         row[f"{metric}_CI_high"] = ci_high

#     # Efficiency ratio
#     row["planning_efficiency_ratio"] = df["planning_time"].mean() / allowed_time

#     summary_rows.append(row)

# summary_df = pd.DataFrame(summary_rows)
# summary_df.sort_values("allowed_time", inplace=True)
# summary_df.to_csv(os.path.join(output_dir, "summary_statistics.csv"), index=False)

# print("✅ Summary statistics saved.")

# # ==============================
# # 🔹 ANOVA + EFFECT SIZE
# # ==============================

# anova_results = []

# metrics = [
#     "success",
#     "planning_time",
#     "wall_time",
#     "path_length_joint",
#     "trajectory_duration",
#     "elapsed_time_sec"
# ]

# for metric in metrics:
#     groups = [data[t][metric] for t in sorted(data.keys())]
#     f_val, p_val = stats.f_oneway(*groups)

#     # Effect size (Eta squared)
#     grand_mean = combined_df[metric].mean()
#     ss_between = sum(
#         len(data[t][metric]) * (data[t][metric].mean() - grand_mean) ** 2
#         for t in data
#     )
#     ss_total = sum((combined_df[metric] - grand_mean) ** 2)
#     eta_squared = ss_between / ss_total

#     anova_results.append({
#         "metric": metric,
#         "F_value": f_val,
#         "p_value": p_val,
#         "eta_squared": eta_squared
#     })

# anova_df = pd.DataFrame(anova_results)
# anova_df.to_csv(os.path.join(output_dir, "anova_results.csv"), index=False)

# print("✅ ANOVA results saved.")

# # ==============================
# # 🔹 REGRESSION ANALYSIS
# # ==============================

# regression_results = []

# for metric in [
#     "success",
#     "planning_time",
#     "path_length_joint",
#     "trajectory_duration"
# ]:
#     X = combined_df["allowed_time"].values.reshape(-1, 1)
#     y = combined_df[metric].values

#     model = LinearRegression()
#     model.fit(X, y)
#     r_squared = model.score(X, y)

#     regression_results.append({
#         "metric": metric,
#         "slope": model.coef_[0],
#         "intercept": model.intercept_,
#         "R_squared": r_squared
#     })

# reg_df = pd.DataFrame(regression_results)
# reg_df.to_csv(os.path.join(output_dir, "regression_results.csv"), index=False)

# print("✅ Regression analysis saved.")

# # ==============================
# # 🔹 PLOTTING SECTION
# # ==============================

# def bar_plot(y_col, ylabel, filename):
#     plt.figure()
#     plt.bar(summary_df["allowed_time"], summary_df[y_col])
#     plt.xlabel("Allowed Planning Time (s)")
#     plt.ylabel(ylabel)
#     plt.title(f"{ylabel} vs Allowed Planning Time")
#     plt.savefig(os.path.join(output_dir, filename), dpi=300)
#     plt.close()

# bar_plot("success_rate_%", "Success Rate (%)", "success_rate.png")
# bar_plot("planning_time_mean", "Mean Planning Time (s)", "planning_time.png")
# bar_plot("path_length_joint_mean", "Mean Path Length", "path_length.png")
# bar_plot("trajectory_duration_mean", "Mean Trajectory Duration (s)", "trajectory_duration.png")
# bar_plot("wall_time_mean", "Mean Wall Time (s)", "wall_time.png")

# # Boxplot
# plt.figure()
# combined_df.boxplot(column="planning_time", by="allowed_time")
# plt.title("Planning Time Distribution")
# plt.suptitle("")
# plt.xlabel("Allowed Planning Time (s)")
# plt.ylabel("Planning Time Used (s)")
# plt.savefig(os.path.join(output_dir, "planning_time_boxplot.png"), dpi=300)
# plt.close()

# print("✅ All plots saved (300 DPI).")
# print("\n🎉 FULL ANALYSIS COMPLETE.")
# print(f"Results stored in folder: {output_dir}")