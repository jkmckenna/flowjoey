#flow_animation

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
        if f'{construct}_kde_2d_values' in adata.uns:
            pass
        else:
            try:
                print(f"Calculation KDE for {construct}")
                construct_subset = adata.obs[adata.obs['Construct'] == construct].copy()

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
            except:
                adata.uns[f'{construct}_kde_2d_values'] = False
                print(f"KDE calculation failed for {construct}. Skipping this construct.")

def anim_flow(adata, construct, x_cat, y_cat, time_per_frame=0.1, num_steps=50, levels=20, outdir='', save_gif=False, save_html=False, xlim=False, ylim=False, zlim=False, plot_3d=False):
    """
    Animate the 2D KDE plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter

    construct_subset = adata.obs[adata.obs['Construct'] == construct]
    time_points = construct_subset['time_cat'].cat.categories

    if time_per_frame:
        # Calculate total frames for scaled interpolation
        total_time = time_points[-1] - time_points[0]
        total_frames = int(total_time / time_per_frame)
        frame_times = np.linspace(time_points[0], time_points[-1], total_frames)

    else:
        total_frames = (len(time_points) - 1) * num_steps

    kde_values, X, Y = adata.uns[f'{construct}_kde_2d_values']

    if xlim:
        x_min, x_max = xlim
    else:
        x_min, x_max = [construct_subset[x_cat].min(), construct_subset[x_cat].max()]

    if ylim:
        y_min, y_max = ylim
    else:
        y_min, y_max = [construct_subset[y_cat].min(), construct_subset[y_cat].max()]

    if zlim:
        z_min, z_max = zlim
    else:
        z_min, z_max = [0, 1.2 * np.max([np.max(k) for k in kde_values])]

    if type(adata.uns[f'{construct}_kde_2d_values']) == list:

        if not plot_3d:
            # Create a 2D figure
            fig, ax = plt.subplots(figsize=(6, 6))  # Square figure dimensions

        else:
            # Create a 3D figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # Initialize plot
        def init():
            """
            Init function for the animation.
            """
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(x_cat)
            ax.set_ylabel(y_cat)
            if plot_3d:
                ax.set_zlabel("Density")
            else:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            ax.set_title(f"{construct} at 0 hours", pad=20)
            return ax,

        if time_per_frame:
            # Interpolation function
            def interpolate_kde(frame_idx):
                current_time = frame_times[frame_idx]
                # Find the two nearest time points
                for i in range(len(time_points) - 1):
                    if time_points[i] <= current_time < time_points[i + 1]:
                        start_idx, end_idx = i, i + 1
                        break
                else:
                    start_idx, end_idx = len(time_points) - 2, len(time_points) - 1  # Edge case for the last frame

                # Linear interpolation between time points
                alpha = (current_time - time_points[start_idx]) / (time_points[end_idx] - time_points[start_idx])
                interpolated_kde = (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]

                return interpolated_kde, current_time
        
        else:
            # Interpolate KDE values for smooth transitions
            def interpolate_kde(frame, num_steps=num_steps):
                """
                Interpolation function
                """
                start_idx = frame // num_steps
                end_idx = (frame // num_steps) + 1
                alpha = (frame % num_steps) / num_steps

                if end_idx >= len(kde_values):
                    end_idx = len(kde_values) - 1

                # Interpolate between two KDE frames
                interpolated_kde = (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]
                interpolated_time = (1 - alpha) * float(time_points[start_idx]) + alpha * float(time_points[end_idx])
                return interpolated_kde, interpolated_time

        def update(frame, num_steps=num_steps, levels=levels):
            """
            Update function for simulating KDE evolution
            """
            if frame < total_frames:  # Forward
                current_frame = frame
            else:  # Reverse
                current_frame = 2 * total_frames - frame

            Z, interpolated_time = interpolate_kde(current_frame)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(x_cat)
            ax.set_ylabel(y_cat)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f"{construct} at {interpolated_time:.1f} hours", pad=20)

            ax.clear()
            if not plot_3d:
                # 2D plotting
                contour = ax.contour(X, Y, Z, levels=levels, cmap="viridis")
                return contour.collections
            else:
                # 3D plotting
                ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.9)
                ax.set_zlim(z_min, z_max)
                ax.set_zlabel("Density")
                return ax,

            
        if plot_3d:
           print(f"Animating 3D KDE for {construct}")
           prefix = 'KDE_3D'
        else: 
            print(f"Animating 2D KDE for {construct}")
            prefix = 'KDE_2D'
        num_frames = (len(time_points) - 1) * num_steps * 2  # Double for forward and reverse
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=40)

        if save_gif:
            # Save animation to output directory as a gif
            output_gif = os.path.join(outdir, f"{prefix}_{construct}_animation.gif")
            ani.save(output_gif, writer=PillowWriter(fps=25))
            print(f"Animation saved to {output_gif}")

        if save_html:
            # Save animation to HTML
            output_html = os.path.join(outdir, f"{prefix}_{construct}_animation.html")
            ani.save(output_html, writer=HTMLWriter(fps=25))
            print(f"Animation saved to {output_html}")

def plot_interactive_3d_kde(adata, construct, x_cat, y_cat):
    """
    Create an interactive 3D KDE plot using Plotly.
    """
    import plotly.graph_objects as go
    
    construct_subset = adata.obs[adata.obs['Construct'] == construct]
    kde_values, X, Y = adata.uns[f'{construct}_kde_2d_values']

    fig = go.Figure()

    for i, Z in enumerate(kde_values):
        fig.add_trace(
            go.Surface(
                z=Z,
                x=X[0],  # X grid values
                y=Y[:, 0],  # Y grid values
                name=f"Time {i}",
                showscale=False,
                opacity=0.8,
            )
        )

    fig.update_layout(
        title=f"Interactive 3D KDE for {construct}",
        scene=dict(
            xaxis_title=x_cat,
            yaxis_title=y_cat,
            zaxis_title="Density",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()