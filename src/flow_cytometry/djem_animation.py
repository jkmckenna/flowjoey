###change some more

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import gaussian_kde
from matplotlib import rcParams
from IPython.display import HTML

# Set embed limit for animations
rcParams['animation.embed_limit'] = 300 * 1024 * 1024  # 300 MB

# Define output directory
output_dir = "/Users/Djem/Desktop/PostdocFiles/Experiments/240517_Exp 61_Vinsons 16 in B12/Python Analyses"

# Filter data for Hif1a_QNtoD, Hif1a_RKtoD, and Hif1a constructs
Hif1a_QNtoD_data = adata.obs[adata.obs['Construct'] == 'Hif1a_QNtoD']
Hif1a_RKtoD_data = adata.obs[adata.obs['Construct'] == 'Hif1a_RKtoD']
Hif1a_data = adata.obs[adata.obs['Construct'] == 'Hif1a']

# Ensure `time_cat` is treated as an ordered categorical variable
for data in [Hif1a_QNtoD_data, Hif1a_RKtoD_data, Hif1a_data]:
    data['time_cat'] = pd.Categorical(
        data['time_cat'], 
        categories=sorted(data['time_cat'].unique()), 
        ordered=True
    )

# Define grid resolution and smoothing parameters
x_grid = np.linspace(Hif1a_QNtoD_data['JFX549_intensity'].min(), Hif1a_QNtoD_data['JFX549_intensity'].max(), 200)
y_grid = np.linspace(Hif1a_QNtoD_data['GFP_intensity'].min(), Hif1a_QNtoD_data['GFP_intensity'].max(), 200)
X, Y = np.meshgrid(x_grid, y_grid)

# Precompute KDE values for each construct
def precompute_kde(data):
    time_points = data['time_cat'].cat.categories
    kde_values = []
    for time_point in time_points:
        subset = data[data['time_cat'] == time_point]
        x, y = subset['JFX549_intensity'], subset['GFP_intensity']
        kde = gaussian_kde(np.vstack([x, y]), bw_method=0.2)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        kde_values.append(Z)
    return time_points, kde_values

time_points_QN, kde_values_QN = precompute_kde(Hif1a_QNtoD_data)
time_points_RK, kde_values_RK = precompute_kde(Hif1a_RKtoD_data)
time_points_Hif1a, kde_values_Hif1a = precompute_kde(Hif1a_data)

# Calculate the number of interpolated frames based on 6-minute intervals
def calculate_frames(time_points):
    times = np.array([float(tp) for tp in time_points])  # Time already in hours
    time_diffs = np.diff(times)
    interpolated_frames = (time_diffs * 10).astype(int)  # 6 minutes is 1/10 of an hour
    return interpolated_frames, times

frames_QN, times_QN = calculate_frames(time_points_QN)
frames_RK, times_RK = calculate_frames(time_points_RK)
frames_Hif1a, times_Hif1a = calculate_frames(time_points_Hif1a)

max_frames = max(np.sum(frames_QN), np.sum(frames_RK), np.sum(frames_Hif1a))

# Create figure for three side-by-side animations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Three side-by-side plots

# Initialize plot
def init():
    for ax, title in zip(axes, ["Hif1a_QNtoD", "Hif1a_RKtoD", "Hif1a"]):
        ax.clear()
        ax.set_xlim(Hif1a_QNtoD_data['JFX549_intensity'].min(), Hif1a_QNtoD_data['JFX549_intensity'].max())
        ax.set_ylim(Hif1a_QNtoD_data['GFP_intensity'].min(), Hif1a_QNtoD_data['GFP_intensity'].max())
        ax.set_xlabel("Halo Intensity")
        ax.set_ylabel("GFP Intensity")
        ax.set_title(f"Construct: {title}, Time: 0 hours")
    return axes

# Interpolate between KDE frames
def interpolate_kde(kde_values, frame, frames):
    cum_frames = np.cumsum(np.insert(frames, 0, 0))
    start_idx = np.searchsorted(cum_frames, frame, side="right") - 1
    if start_idx == len(frames):
        return kde_values[-1]
    alpha = (frame - cum_frames[start_idx]) / frames[start_idx]
    return (1 - alpha) * kde_values[start_idx] + alpha * kde_values[start_idx + 1]

# Compute interpolated time directly from AnnData time points
def compute_time(frame, frames, times):
    cum_frames = np.cumsum(np.insert(frames, 0, 0))
    start_idx = np.searchsorted(cum_frames, frame, side="right") - 1
    if start_idx == len(frames):
        return times[-1]
    alpha = (frame - cum_frames[start_idx]) / frames[start_idx]
    return (1 - alpha) * times[start_idx] + alpha * times[start_idx + 1]

# Update plot for each frame
def update(frame):
    Z_QN = interpolate_kde(kde_values_QN, frame, frames_QN)
    Z_RK = interpolate_kde(kde_values_RK, frame, frames_RK)
    Z_Hif1a = interpolate_kde(kde_values_Hif1a, frame, frames_Hif1a)

    time_QN = compute_time(frame, frames_QN, times_QN)
    time_RK = compute_time(frame, frames_RK, times_RK)
    time_Hif1a = compute_time(frame, frames_Hif1a, times_Hif1a)

    axes[0].clear()
    axes[0].contour(X, Y, Z_QN, levels=75, cmap="viridis")
    axes[0].set_xlim(Hif1a_QNtoD_data['JFX549_intensity'].min(), Hif1a_QNtoD_data['JFX549_intensity'].max())
    axes[0].set_ylim(Hif1a_QNtoD_data['GFP_intensity'].min(), Hif1a_QNtoD_data['GFP_intensity'].max())
    axes[0].set_title(f"Construct: Hif1a_QNtoD, Time: {time_QN:.1f} hours")

    axes[1].clear()
    axes[1].contour(X, Y, Z_RK, levels=75, cmap="viridis")
    axes[1].set_xlim(Hif1a_QNtoD_data['JFX549_intensity'].min(), Hif1a_QNtoD_data['JFX549_intensity'].max())
    axes[1].set_ylim(Hif1a_QNtoD_data['GFP_intensity'].min(), Hif1a_QNtoD_data['GFP_intensity'].max())
    axes[1].set_title(f"Construct: Hif1a_RKtoD, Time: {time_RK:.1f} hours")

    axes[2].clear()
    axes[2].contour(X, Y, Z_Hif1a, levels=75, cmap="viridis")
    axes[2].set_xlim(Hif1a_QNtoD_data['JFX549_intensity'].min(), Hif1a_QNtoD_data['JFX549_intensity'].max())
    axes[2].set_ylim(Hif1a_QNtoD_data['GFP_intensity'].min(), Hif1a_QNtoD_data['GFP_intensity'].max())
    axes[2].set_title(f"Construct: Hif1a, Time: {time_Hif1a:.1f} hours")

    return axes

# Create animation
ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=False, interval=80)

# Save animation
output_file = f"{output_dir}/Hif1a_QNtoD_vs_Hif1a_RKtoD_vs_Hif1a_correct_axes_labels.gif"
ani.save(output_file, writer=PillowWriter(fps=25))
print(f"Animation saved to {output_file}")

# Display the animation inline
#HTML(ani.to_jshtml())
