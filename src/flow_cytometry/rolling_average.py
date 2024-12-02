#rolling_average

def rolling_average(data, intensity_col, target_col, window_size):
    """
    Take a dataframe and roll a window over the intensity column values and calculate the average value within a target column.

    Parameters:
        data (df): Input dataframe
        intensity_col (str): String indicating the column in the dataframe to roll over.
        target_col (str): String indicating the column in the dataframe to extract the mean from.
        window_size (float): The window size to roll with.

    returns:
        data (df): Output dataframe with added rolling average column.
    
    """
    data = data.sort_values(intensity_col).reset_index(drop=True)
    result = []
    for i, center in enumerate(data[intensity_col]):
        lower = center - window_size / 2
        upper = center + window_size / 2
        in_window = data[(data[intensity_col] >= lower) & (data[intensity_col] <= upper)]
        result.append(in_window[target_col].mean())

    data[f"rolling_{target_col}"] = result

    return data