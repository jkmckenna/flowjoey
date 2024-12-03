#flow_animation

def kde_2d_evolution(adata, x_cat, y_cat, constructs=False):
    """
    Calculate the KDE evolution over time for each constuct with adata over two dimensions of interest.

    Parameters:
        adata (AnnData):
        x_cat (str):
        y_cat (str):
        constructs (list of str):

    Returns:
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import gaussian_kde

    if not constructs:
        constructs = adata.obs['Construct'].cat.categories
    else:
        pass

    for construct in constructs:
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

def anim_flow(adata, construct, x_cat, y_cat, time_per_frame=False, num_steps=50, levels=20, outdir='', save_gif=False, save_html=False, xlim=False, ylim=False, zlim=False, plot_3d=False, add_histograms=False):
    """
    Animate the 2D KDE plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
    from matplotlib.gridspec import GridSpec

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

    # Set axis limits
    x_min, x_max = xlim if xlim else (construct_subset[x_cat].min(), construct_subset[x_cat].max())
    y_min, y_max = ylim if ylim else (construct_subset[y_cat].min(), construct_subset[y_cat].max())
    z_min, z_max = zlim if zlim else (0, 1.2 * np.max([np.max(k) for k in kde_values]))

    if type(adata.uns[f'{construct}_kde_2d_values']) == list:

        # Create figure
        if add_histograms and not plot_3d:
            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(4, 4, figure=fig)
            ax_main = fig.add_subplot(gs[1:4, 0:3])  # Main KDE plot
            ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)  # X-axis histogram
            ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)  # Y-axis histogram
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)
        else:
            fig = plt.figure(figsize=(10, 8)) if plot_3d else plt.figure(figsize=(6, 6))
            ax_main = fig.add_subplot(111, projection='3d') if plot_3d else fig.add_subplot(111)

        # Initialize plot
        def init():
            """
            Init function for the animation.
            """
            ax_main.clear()
            if add_histograms and not plot_3d:
                ax_top.clear()
                ax_right.clear()
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
            ax_main.set_xlabel(x_cat)
            ax_main.set_ylabel(y_cat)
            if plot_3d:
                ax_main.set_zlabel("Density")
            else:
                ax_main.spines['top'].set_visible(False)
                ax_main.spines['right'].set_visible(False)
            ax_main.set_title(f"{construct} at 0 hours", pad=20)
            return ax_main,

        # Interpolation function
        def interpolate_kde(frame_idx):
            """
            Interpolation function
            """
            if time_per_frame:
                current_time = frame_times[frame_idx]
                for i in range(len(time_points) - 1):
                    if time_points[i] <= current_time < time_points[i + 1]:
                        start_idx, end_idx = i, i + 1
                        break
                else:
                    start_idx, end_idx = len(time_points) - 2, len(time_points) - 1
                alpha = (current_time - time_points[start_idx]) / (time_points[end_idx] - time_points[start_idx])
            else:
                start_idx = frame_idx // num_steps
                end_idx = min(start_idx + 1, len(kde_values) - 1)
                alpha = (frame_idx % num_steps) / num_steps
                current_time = (1 - alpha) * float(time_points[start_idx]) + alpha * float(time_points[end_idx])

            interpolated_kde = (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]
            return interpolated_kde, current_time

        def update(frame, levels=levels):
            """
            Update function for simulating KDE evolution
            """
            ax_main.clear()
            if add_histograms and not plot_3d:
                ax_top.clear()
                ax_right.clear()

            if frame < total_frames:  # Forward
                current_frame = frame
            else:  # Reverse
                current_frame = 2 * total_frames - frame - 1

            Z, interpolated_time = interpolate_kde(current_frame)

            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
            ax_main.set_xlabel(x_cat)
            ax_main.set_ylabel(y_cat)
            ax_main.set_title(f"{construct} at {interpolated_time:.1f} hours", pad=20)

            if plot_3d:
                # 3D plotting
                ax_main.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.9)
                ax_main.set_zlim(z_min, z_max)
                ax_main.set_zlabel("Density")
                return ax_main,
    
            else:
                # 2D plotting
                ax_main.spines['top'].set_visible(False)
                ax_main.spines['right'].set_visible(False)
                contour = ax_main.contour(X, Y, Z, levels=levels, cmap="viridis")

                if add_histograms:
                    # Update histograms
                    current_data = construct_subset[
                        construct_subset['time_cat'] == time_points[current_frame // num_steps]
                    ]
                    ax_top.hist(current_data[x_cat], bins=30, color="blue", alpha=0.6)
                    ax_top.set_ylabel("Count")

                    ax_right.hist(current_data[y_cat], bins=30, color="green", alpha=0.6, orientation="horizontal")
                    ax_right.set_xlabel("Count")

                return contour.collections
            
        if plot_3d:
           print(f"Animating 3D KDE for {construct}")
           prefix = 'KDE_3D'
        else: 
            print(f"Animating 2D KDE for {construct}")
            prefix = 'KDE_2D'

        num_frames = total_frames * 2  # Double for forward and reverse
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
    import plotly.io as pio
    pio.renderers.default = "notebook"
    
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


def plotly_animate_kde(adata, construct, x_cat, y_cat):
    """
    Create an interactive 3D animation of KDE evolution using Plotly.
    """
    import plotly.graph_objects as go
    import numpy as np
    construct_subset = adata.obs[adata.obs['Construct'] == construct].copy()
    kde_values, X, Y = adata.uns[f'{construct}_kde_2d_values']
    time_points = construct_subset['time_cat'].cat.categories.astype(float)  # Ensure time points are numeric

    # Create the base figure
    fig = go.Figure()

    # Add the first frame (initial state)
    Z_initial = kde_values[0]
    fig.add_trace(
        go.Surface(
            z=Z_initial,
            x=X[0],  # X grid values
            y=Y[:, 0],  # Y grid values
            colorscale="Viridis",
            name=f"Time {time_points[0]}",
            showscale=False,
        )
    )

    # Add frames for animation
    frames = []
    for i, Z in enumerate(kde_values):
        frame = go.Frame(
            data=[
                go.Surface(
                    z=Z,
                    x=X[0],
                    y=Y[:, 0],
                    colorscale="Viridis",
                    showscale=False,
                )
            ],
            name=f"Time {time_points[i]:.1f}",
        )
        frames.append(frame)

    fig.frames = frames

    # Add sliders and animation controls
    fig.update_layout(
        title=f"3D KDE Animation for {construct}",
        scene=dict(
            xaxis_title=x_cat,
            yaxis_title=y_cat,
            zaxis_title="Density",
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [[f"Time {time_points[i]:.1f}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{time_points[i]:.1f}",
                        "method": "animate",
                    }
                    for i in range(len(time_points))
                ],
                "currentvalue": {"font": {"size": 20}, "prefix": "Time: ", "visible": True, "xanchor": "center"},
                "transition": {"duration": 300, "easing": "cubic-in-out"},
            }
        ],
    )

    fig.show()
