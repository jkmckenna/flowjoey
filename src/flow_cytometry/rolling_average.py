#rolling_average

def rolling_average(adata, groupby, intensity_col, target_col, window_size, constructs=False):
    """
    Take a dataframe and roll a window over the intensity column values and calculate the average value within a target column.

    Parameters:
        adata (AnnData): Input AnnData
        groupby (list of str): List of strings representing categorical columns of the data to use for grouping. Only works when two cols are passed currently.
        intensity_col (str): String indicating the column in the dataframe to roll over.
        target_col (str): String indicating the column in the dataframe to extract the mean from.
        window_size (float): The window size to roll with.
        constructs (list of str): List of construct names to specifically iterate over. If False, just uses all constructs

    returns:
        adata
    
    """
    from tqdm import tqdm

    construct_col, time_col = groupby

    if constructs:
        subset_mask = adata.obs[construct_col].isin(constructs)
        subset = adata[subset_mask].copy()
        grouped = subset.obs.groupby(groupby)
    else:
        grouped = adata.obs.groupby(groupby)

    for (construct, time_cat), group in tqdm(grouped, desc="Processing Groups", total=len(grouped)):
        if f"{construct}_{time_cat}" in adata.uns:
            pass
        else:
            group = group.sort_values(intensity_col).reset_index(drop=True)
            result = []
            for i, center in enumerate(group[intensity_col]):
                lower = center - window_size / 2
                upper = center + window_size / 2
                in_window = group[(group[intensity_col] >= lower) & (group[intensity_col] <= upper)]
                result.append(in_window[target_col].mean())

            group[f"rolling_{target_col}"] = result

            adata.uns[f"{construct}_{time_cat}"] = group[[intensity_col, f'rolling_{target_col}']]
