#flow_animation

## Calculates KDEs at each time point for each construct and saves it into the AnnData ##
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
    from tqdm import tqdm

    if not constructs:
        constructs = adata.obs['Construct'].cat.categories
    else:
        pass

    for construct in constructs:
        if f'{construct}_kde_2d_values' in adata.uns:
            pass
        else:
            try:
                print(f"Calculating KDE for {construct}")
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

## 2D Plotting Functions ##
def anim_flow_combined_v2(
    adata, construct, x_cat, y_cat, time_per_frame=False, num_steps=50, levels=20,
    outdir='', save_gif=False, save_html=False, xlim=None, ylim=None, zlim=None,
    plot_3d=False, add_histograms=False
):
    """
    Animate the 2D/3D KDE plot with an updated progress bar.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
    from matplotlib.gridspec import GridSpec
    from tqdm import tqdm

    construct_subset = adata.obs[adata.obs['Construct'] == construct]
    time_points = construct_subset['time_cat'].cat.categories.astype(float)

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
        prefix = 'KDE_3D' if plot_3d else 'KDE_2D'
        fig = plt.figure(figsize=(12, 10)) if add_histograms and not plot_3d else plt.figure(figsize=(10, 8))
        
        if add_histograms and not plot_3d:
            gs = GridSpec(5, 5, figure=fig)
            ax_main = fig.add_subplot(gs[1:5, 0:4])  # Main KDE plot
            ax_top = fig.add_subplot(gs[0, 0:4], sharex=ax_main)  # X-axis histogram
            ax_right = fig.add_subplot(gs[1:5, 4], sharey=ax_main)  # Y-axis histogram
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)
        else:
            ax_main = fig.add_subplot(111, projection='3d') if plot_3d else fig.add_subplot(111)

        # Precompute interpolations to avoid repeated computation
        interpolated_kdes = []
        interpolated_times = []
        max_Y_density = 0
        max_X_density = 0

        if time_per_frame:
            frame_indices = [np.searchsorted(time_points, t) for t in frame_times]
            for t_idx, t in zip(frame_indices, frame_times):
                start_idx = t_idx - 1 if t_idx > 0 else 0
                end_idx = min(t_idx, len(kde_values) - 1)
                # Avoid division by zero
                if start_idx == end_idx or time_points[end_idx] == time_points[start_idx]:
                    Z = kde_values[start_idx]  # Use the KDE of the single time point
                else:
                    alpha = (t - time_points[start_idx]) / (time_points[end_idx] - time_points[start_idx])
                    Z = (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]

                interpolated_kdes.append(Z)
                interpolated_times.append(t)
                y_density = np.max(np.sum(Z, axis=1))
                x_density = np.max(np.sum(Z, axis=0))
                if y_density > max_Y_density:
                    max_Y_density = y_density
                if x_density > max_X_density:
                    max_X_density = x_density                
            

        else:
            for frame_idx in range(total_frames):
                start_idx = frame_idx // num_steps
                end_idx = min(start_idx + 1, len(kde_values) - 1)
                alpha = (frame_idx % num_steps) / num_steps
                Z = (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]
                t = (1 - alpha) * float(time_points[start_idx]) + alpha * float(time_points[end_idx])
                interpolated_kdes.append(Z)
                interpolated_times.append(t)
                y_density = np.max(np.sum(Z, axis=1))
                x_density = np.max(np.sum(Z, axis=0))
                if y_density > max_Y_density:
                    max_Y_density = y_density
                if x_density > max_X_density:
                    max_X_density = x_density  

        # Initialize and run animation with manual progress bar
        pbar = tqdm(total=total_frames, desc="Rendering Frames")

        def update(frame):
            ax_main.clear()
            if add_histograms and not plot_3d:
                ax_top.clear()
                ax_right.clear()

            Z = interpolated_kdes[frame]
            interpolated_time = interpolated_times[frame]

            # Update main KDE plot
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
            ax_main.set_xlabel(x_cat)
            ax_main.set_ylabel(y_cat)
            ax_main.set_title(f"{construct} at {interpolated_time:.1f} hours", pad=20)
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)

            if plot_3d:
                ax_main.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.9)
                ax_main.set_zlim(z_min, z_max)
                ax_main.set_zlabel("Density")
                pbar.update(1)
                return ax_main,
            else:
                contour = ax_main.contour(X, Y, Z, levels=levels, cmap="viridis")

                if add_histograms:
                    # Marginal histograms from KDE
                    top_density = np.sum(Z, axis=0)
                    top_density_normalized = top_density / np.sum(top_density)
                    cumulative_top_density = np.cumsum(top_density_normalized)
                    ax_top.bar(X[0, :], top_density, width=np.diff(X[0, :])[0], color="blue", alpha=0.6)
                    ax_top.set_ylabel("Density")
                    ax_top.set_title(f"{construct} KDE at {interpolated_time:.1f} hours", fontsize=14, pad=20)
                    ax_top.set_ylim(0, max_X_density)
                    
                    top_median_index = np.searchsorted(cumulative_top_density, 0.5)
                    median_top_position = X[0, :][top_median_index]
                    ax_top.axvline(median_top_position, color='red', linestyle='-', linewidth=1.5)
                    ax_top.text(median_top_position, max_X_density*1, f'Median: {median_top_position:.2f}', color='red', ha="center", va="bottom")

                    right_density = np.sum(Z, axis=1)
                    right_density_normalized = right_density / np.sum(right_density)
                    cumulative_right_density = np.cumsum(right_density_normalized)
                    ax_right.barh(Y[:, 0], right_density, height=np.diff(Y[:, 0])[0], color="green", alpha=0.6)
                    ax_right.set_xlabel("Density")
                    ax_right.set_xlim(0, max_Y_density)
                    
                    right_median_index = np.searchsorted(cumulative_right_density, 0.5)
                    median_right_position = Y[:, 0][right_median_index]
                    ax_right.axhline(median_right_position, color='red', linestyle='-', linewidth=1.5)
                    ax_right.text(max_Y_density*0.5, median_right_position*1.05, f'Median: {median_right_position:.0f}', color='red', ha="center", va="bottom")

                    ax_top.spines['top'].set_visible(False)
                    ax_top.spines['right'].set_visible(False)
                    ax_top.spines['left'].set_visible(False)
                    ax_right.spines['top'].set_visible(False)
                    ax_right.spines['right'].set_visible(False)
                    ax_right.spines['left'].set_visible(False)

                    plt.setp(ax_top.get_xticklabels(), visible=False)
                    plt.setp(ax_right.get_yticklabels(), visible=False)

                pbar.update(1)
                return contour.collections

        ani = FuncAnimation(fig, update, frames=total_frames, blit=False, interval=40)

        pbar.close()

        if save_gif:
            output_gif = os.path.join(outdir, f"{prefix}_{construct}_animation.gif")
            ani.save(output_gif, writer=PillowWriter(fps=25))
            print(f"Animation saved to {output_gif}")

        if save_html:
            output_html = os.path.join(outdir, f"{prefix}_{construct}_animation.html")
            ani.save(output_html, writer=HTMLWriter(fps=25))
            print(f"Animation saved to {output_html}")

        return ani


## 3d Plotting Functions ##
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

def plotly_animate_kde_with_interpolation(adata, construct, x_cat, y_cat, num_steps=10):
    """
    Create an interactive 3D animation of KDE evolution using Plotly with interpolation.
    """
    import plotly.graph_objects as go
    import numpy as np
    construct_subset = adata.obs[adata.obs['Construct'] == construct]
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

    # Generate frames with interpolation
    frames = []
    for i in range(len(time_points) - 1):
        start_idx, end_idx = i, i + 1
        for step in range(num_steps):
            # Linear interpolation
            alpha = step / num_steps
            Z_interpolated = (1 - alpha) * kde_values[start_idx] + alpha * kde_values[end_idx]
            interpolated_time = (1 - alpha) * time_points[start_idx] + alpha * time_points[end_idx]

            frame = go.Frame(
                data=[
                    go.Surface(
                        z=Z_interpolated,
                        x=X[0],
                        y=Y[:, 0],
                        colorscale="Viridis",
                        showscale=False,
                    )
                ],
                name=f"Time {interpolated_time:.1f}",
            )
            frames.append(frame)

    # Add the last frame
    Z_last = kde_values[-1]
    frames.append(
        go.Frame(
            data=[
                go.Surface(
                    z=Z_last,
                    x=X[0],
                    y=Y[:, 0],
                    colorscale="Viridis",
                    showscale=False,
                )
            ],
            name=f"Time {time_points[-1]}",
        )
    )

    fig.frames = frames

    # Add sliders and animation controls
    fig.update_layout(
        title=f"3D KDE Animation for {construct} with Interpolation",
        scene=dict(
            xaxis_title=x_cat,
            yaxis_title=y_cat,
            zaxis_title="Density",
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
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
                        "args": [[f"Time {step.name.split()[1]}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": step.name.split()[1],
                        "method": "animate",
                    }
                    for step in frames
                ],
                "currentvalue": {"font": {"size": 20}, "prefix": "Time: ", "visible": True, "xanchor": "center"},
                "transition": {"duration": 300, "easing": "cubic-in-out"},
            }
        ],
    )

    fig.show()

#### Archived flow_animation functions ####

## Archived 2D functions ##

## Archived 3D functions ##

def anim_flow_2D_v1(adata, construct, x_cat, y_cat, time_per_frame=False, num_steps=50, levels=20, outdir='', save_gif=False, save_html=False, xlim=False, ylim=False, add_histograms=False):
    """
    Animate the 2D KDE plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
    from matplotlib.gridspec import GridSpec
    from tqdm import tqdm

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

    if type(adata.uns[f'{construct}_kde_2d_values']) == list:
        print(f"Animating 2D KDE for {construct}")
        prefix = 'KDE_2D'
        # Create figure
        if add_histograms:
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(5, 5, figure=fig)
            ax_main = fig.add_subplot(gs[1:5, 0:4])  # Main KDE plot
            ax_top = fig.add_subplot(gs[0, 0:4], sharex=ax_main)  # X-axis histogram
            ax_right = fig.add_subplot(gs[1:5, 4], sharey=ax_main)  # Y-axis histogram
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)
        else:
            fig = plt.figure(figsize=(6, 6))
            ax_main = fig.add_subplot(111)

        # Initialize plot
        def init():
            """
            Init function for the animation.
            """
            ax_main.clear()
            if add_histograms:
                ax_top.clear()
                ax_right.clear()
                ax_top.spines['top'].set_visible(False)
                ax_top.spines['right'].set_visible(False)
                ax_top.spines['left'].set_visible(False)
                ax_right.spines['top'].set_visible(False)
                ax_right.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
            ax_main.set_xlabel(x_cat)
            ax_main.set_ylabel(y_cat)
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            ax_main.set_title(f"{construct} at 0 hours", pad=20)
            return ax_main,

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

        pbar = tqdm(total=total_frames, desc="Rendering Frames")

        def update(frame, levels=levels):
            """
            Update function for simulating KDE evolution
            """
            ax_main.clear()
            if add_histograms:
                ax_top.clear()
                ax_right.clear()
                ax_top.spines['top'].set_visible(False)
                ax_top.spines['right'].set_visible(False)
                ax_top.spines['left'].set_visible(False)
                ax_right.spines['top'].set_visible(False)
                ax_right.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)

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

            # 2D plotting
            ax_main.spines['top'].set_visible(False)
            ax_main.spines['right'].set_visible(False)
            contour = ax_main.contour(X, Y, Z, levels=levels, cmap="viridis")

            if add_histograms:
                # Marginal histograms from KDE
                ax_top.bar(X[0, :], np.sum(Z, axis=0), width=np.diff(X[0, :])[0], color="blue", alpha=0.6)
                ax_top.set_ylabel("Density")
                ax_top.set_title(f"{construct} KDE at {interpolated_time:.1f} hours", fontsize=14, pad=20)

                ax_right.barh(Y[:, 0], np.sum(Z, axis=1), height=np.diff(Y[:, 0])[0], color="green", alpha=0.6)
                ax_right.set_xlabel("Density")

            pbar.update(1)  # Update progress bar
            return contour.collections


        num_frames = total_frames * 2  # Double for forward and reverse
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=40)
        pbar.close()  # Close progress bar

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

def anim_flow_3D_v1(adata, construct, x_cat, y_cat, time_per_frame=False, num_steps=50, levels=20, outdir='', save_gif=False, save_html=False, xlim=False, ylim=False, zlim=False):
    """
    Animate the 3D KDE plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
    from tqdm import tqdm

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
        print(f"Animating 3D KDE for {construct}")
        prefix = 'KDE_3D'

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax_main = fig.add_subplot(111, projection='3d')

        # Initialize plot
        def init():
            """
            Init function for the animation.
            """
            ax_main.clear()
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
            ax_main.set_xlabel(x_cat)
            ax_main.set_ylabel(y_cat)
            ax_main.set_zlabel("Density")
            ax_main.set_title(f"{construct} at 0 hours", pad=20)
            return ax_main,

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

        pbar = tqdm(total=total_frames, desc="Rendering Frames")

        def update(frame):
            """
            Update function for simulating KDE evolution
            """
            ax_main.clear()

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

            # 3D plotting
            ax_main.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.9)
            ax_main.set_zlim(z_min, z_max)
            ax_main.set_zlabel("Density")
            pbar.update(1)  # Update progress bar
            return ax_main,

        num_frames = total_frames * 2  # Double for forward and reverse
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=40)
        pbar.close()  # Close progress bar

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

def anim_flow_combined_v1(adata, construct, x_cat, y_cat, time_per_frame=False, num_steps=50, levels=20, outdir='', save_gif=False, save_html=False, xlim=False, ylim=False, zlim=False, plot_3d=False, add_histograms=False):
    """
    Animate the 2D KDE plot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
    from matplotlib.gridspec import GridSpec
    from tqdm import tqdm

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
        if plot_3d:
           print(f"Animating 3D KDE for {construct}")
           prefix = 'KDE_3D'
        else: 
            print(f"Animating 2D KDE for {construct}")
            prefix = 'KDE_2D'
        # Create figure
        if add_histograms and not plot_3d:
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(5, 5, figure=fig)
            ax_main = fig.add_subplot(gs[1:5, 0:4])  # Main KDE plot
            ax_top = fig.add_subplot(gs[0, 0:4], sharex=ax_main)  # X-axis histogram
            ax_right = fig.add_subplot(gs[1:5, 4], sharey=ax_main)  # Y-axis histogram
            plt.setp(ax_top.get_xticklabels(), visible=False)
            plt.setp(ax_right.get_yticklabels(), visible=False)
        else:
            fig = plt.figure(figsize=(10, 8)) if plot_3d else plt.figure(figsize=(6, 6))
            ax_main = fig.add_subplot(111, projection='3d') if plot_3d else fig.add_subplot(111)

        # Initialize progress bar
        pbar = tqdm(total=total_frames * 2, desc="Rendering Frames")

        # Initialize plot
        def init():
            """
            Init function for the animation.
            """
            ax_main.clear()
            if add_histograms and not plot_3d:
                ax_top.clear()
                ax_right.clear()
                ax_top.spines['top'].set_visible(False)
                ax_top.spines['right'].set_visible(False)
                ax_top.spines['left'].set_visible(False)
                ax_right.spines['top'].set_visible(False)
                ax_right.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)
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
                ax_top.spines['top'].set_visible(False)
                ax_top.spines['right'].set_visible(False)
                ax_top.spines['left'].set_visible(False)
                ax_right.spines['top'].set_visible(False)
                ax_right.spines['right'].set_visible(False)
                ax_right.spines['left'].set_visible(False)

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
                pbar.update(1)  # Update progress bar
                return ax_main,
    
            else:
                # 2D plotting
                ax_main.spines['top'].set_visible(False)
                ax_main.spines['right'].set_visible(False)
                contour = ax_main.contour(X, Y, Z, levels=levels, cmap="viridis")

                if add_histograms:
                    # Marginal histograms from KDE
                    ax_top.bar(X[0, :], np.sum(Z, axis=0), width=np.diff(X[0, :])[0], color="blue", alpha=0.6)
                    ax_top.set_ylabel("Density")
                    ax_top.set_title(f"{construct} KDE at {interpolated_time:.1f} hours", fontsize=14, pad=20)

                    ax_right.barh(Y[:, 0], np.sum(Z, axis=1), height=np.diff(Y[:, 0])[0], color="green", alpha=0.6)
                    ax_right.set_xlabel("Density")

                pbar.update(1)  # Update progress bar
                return contour.collections

        num_frames = total_frames * 2  # Double for forward and reverse
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=40)
        pbar.close()  # Close progress bar

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