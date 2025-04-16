import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Clean academic style
plt.style.use('seaborn-v0_8-paper')
rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})

# Data
algorithms = ['Back-and-forth', 'AS', 'MMAS']
metrics = ['Distance (m)', 'Turns (deg)', 'Energy cost (kJ)']

dual = {
    'Distance (m)': [1330.2, 1124.8, 1126.0],
    'Turns (deg)': [1441.2, 1484.2, 1488.9],
    'Energy cost (kJ)': [179.8, 156.6, 156.8]
}

single = {
    'Distance (m)': [1246.9, 1140.6, 1190.9],
    'Turns (deg)': [1131.1, 1278.7, 1041.8],
    'Energy cost (kJ)': [164.7, 154.9, 156.6]
}

colors = ['#4C72B0', '#55A868']  # Dual (blue), Single (green)
bar_width = 0.35
x = np.arange(len(algorithms))

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

for i, metric in enumerate(metrics):
    ax = axes[i]
    dual_vals = dual[metric]
    single_vals = single[metric]

    bars1 = ax.bar(x - bar_width/2, dual_vals, width=bar_width, label='Dual-drone', color=colors[0])
    bars2 = ax.bar(x + bar_width/2, single_vals, width=bar_width, label='Single-drone', color=colors[1])

    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=0)

    # Restore all box borders
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Dynamic y-axis margin
    all_vals = dual_vals + single_vals
    margin = (max(all_vals) - min(all_vals)) * 0.1
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)

    # Annotate bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    if i == 0:
        ax.legend(frameon=True, loc='upper right')

#fig.suptitle('Comparison of Coverage Planning Algorithms\n(Single vs Dual Drone)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()