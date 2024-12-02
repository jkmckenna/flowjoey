#flow_animation

def interpolate_kde(kde_values, frame, num_steps):
    """
    Interpolation function for the KDE grid.
    """
    start_idx = frame // num_steps
    end_idx = (frame // num_steps) + 1
    alpha = (frame % num_steps) / num_steps

    # Clamp to avoid overflow
    if end_idx >= len(kde_values):
        end_idx = len(kde_values) - 1

    return (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]

def update(adata, construct, frame, time_points, num_steps=50, levels=20):
    """
    Update function for simulating KDE evolution
    """
    kde_values = adata.uns[f'{construct}_kde_2d_values']
    construct_data = adata[adata.obs['Construct'] == construct]

    total_frames = (len(time_points) - 1) * num_steps
    if frame < total_frames:  # Forward
        current_frame = frame
    else:  # Reverse
        current_frame = 2 * total_frames - frame

    Z = interpolate_kde(kde_values, frame, num_steps)
    ax.clear()
    ax.contour(X, Y, Z, levels=levels, cmap="viridis")
    ax.set_xlim(construct_data[x_cat].min(), construct_data[x_cat].max())
    ax.set_ylim(construct_data[y_cat].min(), construct_data[y_cat].max())
    current_time = time_points[current_frame // num_steps] if current_frame < len(time_points) * num_steps else time_points[-1]
    ax.set_title(f"{construct} at {current_time} hours")
    return ax,

def kde_2d_evolution(adata, x_cat, y_cat):
    """
    Calculate the KDE evolution over time for each constuct with adata over two dimensions of interest.

    Parameters:
        adata (AnnData):
        x_cat (str):
        y_cat (str):

    Returns:
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import gaussian_kde

    for construct in adata.obs['Construct'].cat.categories:
        print(f"Calculation KDE for {construct}")
        construct_subset = adata.obs[adata.obs['Construct'] == construct]

        # Ensure `time_cat` is treated as an ordered categorical variable
        construct_subset['time_cat'] = pd.Categorical(
            construct_subset['time_cat'], 
            categories=sorted(construct_subset['time_cat'].unique()), 
            ordered=True
        )

        # Define grid resolution and smoothing parameters
        x_grid = np.linspace(construct_subset[x_cat].min(), construct_subset[x_cat].max(), 200)
        y_grid = np.linspace(construct_subset[y_cat].min(), construct_subset[y_cat].max(), 200)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Precompute KDE values for each time point
        time_points = construct_subset['time_cat'].cat.categories
        kde_values = []

        for time_point in time_points:
            time_subset = construct_subset[construct_subset['time_cat'] == time_point]
            x, y = time_subset[x_cat], time_subset[y_cat]
            kde = gaussian_kde(np.vstack([x, y]), bw_method=0.2)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            kde_values.append(Z)

        adata.uns[f'{construct}_kde_2d_values'] = [kde_values, X, Y]

def anim_flow(adata, construct, x_cat, y_cat, num_steps, outdir='', save=False):
    """
    Animate the 2D KDE plot.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    construct_subset = adata[adata.obs['Construct'] == construct]
    time_points = construct_subset['time_cat'].cat.categories

    # Create a figure
    fig, ax = plt.subplots(figsize=(6, 6))  # Square figure dimensions

    # Initialize plot
    def init():
        """
        Init function for the animation.
        """
        ax.clear()
        ax.set_xlim(construct_subset[x_cat].min(), construct_subset[x_cat].max())
        ax.set_ylim(construct_subset[y_cat].min(), construct_subset[y_cat].max())
        ax.set_xlabel(x_cat)
        ax.set_ylabel(y_cat)
        ax.set_title(f"{construct} at 0 hours")
        return ax,


    print(f"Animating 2D KDE for {construct}")
    num_frames = (len(time_points) - 1) * num_steps * 2  # Double for forward and reverse
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=40)

    if save:
        # Save animation to output directory as a single file
        output_file = os.path.join(outdir, f"{construct}_animation.gif")
        ani.save(output_file, writer=PillowWriter(fps=25))
        print(f"Animation saved to {output_file}")