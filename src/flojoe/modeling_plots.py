# modeling_plots

def rolling_average_plot(adata, groupby, intensity_col, target_col, window_size, save_roll=False, outdir='', constructs=False):
    """
    Plots the rolling average of a target column over an intensity column, grouped by constructs and timepoints.

    Parameters:
    - adata: AnnData object containing the data.
    - groupby: Tuple of two columns to group by (construct_col, time_col).
    - intensity_col: Column for the x-axis (e.g., intensity).
    - target_col: Column for calculating the rolling average.
    - window_size: Size of the rolling window.
    - save_roll: Whether to save the plots or display them.
    - outdir: Directory to save plots if save_roll is True.
    - constructs: (list of str): List of construct names to specifically iterate over. If False, just uses all constructs

    # Note: need to add plotting param dict as input for xlim and ylim, and figsize
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    construct_col, time_col = groupby

    if constructs:
        pass
    else:
        constructs = adata.obs[construct_col].cat.categories

    for construct in constructs:
        plt.figure(figsize=(10, 4))
        plt.title(f'Rolling average of {target_col} (Window={window_size})\nConstruct: {construct}', pad=20)
        plt.xlabel(intensity_col)
        plt.ylim(0,1)
        plt.xlim(2.5,4.5)
        plt.ylabel(target_col)
        plt.grid(False)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for time_cat in adata.obs[time_col].cat.categories:
            proportion_pos = adata.obs[(adata.obs[construct_col] == construct) & (adata.obs[time_col] == time_cat)][target_col].mean()
            try:
                df = adata.uns[f"{construct}_{time_cat}"]
                plt.plot(
                    np.array(df[intensity_col]),
                    np.array(df[f'rolling_{target_col}']),
                    label=f'{time_cat} hrs with {proportion_pos:.2f} proportion GFP+'
                )
            except KeyError as e:
                print(f"Error accessing data for {construct}_{time_cat}: {e}")
                continue                
        plt.legend(
            title='Timepoint', 
            loc='center left', 
            bbox_to_anchor=(1, 0.7), 
            frameon=False, 
            fontsize=10, 
            title_fontsize=12
        )

        # Adjust layout to prevent squishing
        plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust rect to allow extra space for the legend
        if save_roll:
            save_name = f'Rolling_Average_{construct}.png'
            outpath = os.path.join(outdir, save_name)
            plt.savefig(outpath, bbox_inches='tight', dpi=300)  # Increased DPI for higher quality
            plt.close()
        else:
            plt.show()