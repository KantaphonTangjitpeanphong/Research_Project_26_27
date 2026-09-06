"""
Combine all six summary_results_*.csv files and plot energy vs model scale
for CPU vs GPU — this is the chart that directly answers the research question.

Run this after all six run_*.py scripts have completed their full grid runs
and produced their summary_results_*.csv files in the same folder.
"""

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load and combine all six summary files
# ------------------------------------------------------------
files = [
    "summary_results_small_cpu.csv", "summary_results_small_gpu.csv",
    "summary_results_medium_cpu.csv", "summary_results_medium_gpu.csv",
    "summary_results_large_cpu.csv", "summary_results_large_gpu.csv", "emission_small_cuda.csv"
    ]

dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f))
    except FileNotFoundError:
        print(f"WARNING: {f} not found — skipping. Run that script's full grid first.")

df = pd.concat(dfs, ignore_index=True)

# Keep model scales in a sensible order for plotting, not alphabetical
order = ["small", "medium", "large"]
df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
df = df.sort_values(["model", "device"])

print("\nCombined summary table:")
print(df.to_string(index=False))
df.to_csv("combined_summary.csv", index=False)
print("\nSaved combined_summary.csv")

# ------------------------------------------------------------
# 2. Plot: energy per trial vs model scale, CPU vs GPU
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for device, group in df.groupby("device"):
    ax.errorbar(
        group["model"], group["mean_energy_kwh"], yerr=group["std_energy_kwh"],
        marker="o", capsize=4, label=device.upper(),
    )
ax.set_xlabel("Model scale")
ax.set_ylabel("Mean energy per trial (kWh)")
ax.set_title("Energy consumption: CPU vs GPU across model scale")
ax.legend()
plt.tight_layout()
plt.savefig("energy_vs_scale.png", dpi=150)
print("Saved energy_vs_scale.png")

# ------------------------------------------------------------
# 3. Plot: estimated cost vs model scale, CPU vs GPU
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for device, group in df.groupby("device"):
    ax.plot(group["model"], group["est_cost_thb"], marker="o", label=device.upper())
ax.set_xlabel("Model scale")
ax.set_ylabel("Estimated cost per trial (THB)")
ax.set_title("Cost: CPU vs GPU across model scale")
ax.legend()
plt.tight_layout()
plt.savefig("cost_vs_scale.png", dpi=150)
print("Saved cost_vs_scale.png")

# ------------------------------------------------------------
# 4. Plot: time vs model scale (useful for the "speed" side of the story)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for device, group in df.groupby("device"):
    ax.plot(group["model"], group["mean_time_sec"] / 60, marker="o", label=device.upper())
ax.set_xlabel("Model scale")
ax.set_ylabel("Mean training time per trial (minutes)")
ax.set_title("Training time: CPU vs GPU across model scale")
ax.legend()
plt.tight_layout()
plt.savefig("time_vs_scale.png", dpi=150)
print("Saved time_vs_scale.png")

print("\nDone. Open the three PNG files to see the charts, or combined_summary.csv for the raw numbers.")
