#load_plate

def load_plate(datadir, transformation_map={'hlog': ['GFP', 'JFX549']}):
    """
    Loads a FCPlate object from a directory of FCS files.

    Parameters:
        datadir (str): String indicating the path to the root directory in which to directly search for FCS files.
        transformation_map (dict): Dictionary mapping transformation types to a list of channels to apply them to.

    Returns:
        plate (FCPlate object): An FCPlate object containing all of the FCS data.
    """
    import os
    from FlowCytometryTools import FCMeasurement, FCPlate

    datadir_basename = os.path.basename(datadir)
    print(f"Loading FCPlate object from FCS files contained within {datadir}")

    files = os.listdir(datadir)
    fcs_files = [file for file in files if '.fcs' in file]
    fcs_paths = [os.path.join(datadir, file) for file in fcs_files]

    # Init a dictionary to hold samples
    sample_dict = {}

    # Load the dictionary with sample IDs which point to the respective FCSMeasurement object
    for file in fcs_paths:
        temp = None
        temp = FCMeasurement(ID='temp', datafile=file)
        temp.ID = temp.meta['WELL ID']
        sample_dict[temp.ID] = temp

    # Load the plate from the dictionary
    plate = FCPlate(datadir_basename, sample_dict, 'name', shape=(8,12))

    # Apply transforms to channels of interest (generally fluorescent channels)
    for transform_type, channels_to_transform in transformation_map.items():
        try:
            plate = plate.transform(transform_type, channels=channels_to_transform)
        except:
            ch = " ".join(channels_to_transform)
            print(f'{transform_type} transform of {ch} experienced an error. Please check that the transform type is valid and that the channel name is present')

    print(plate)

    return plate

def load_plates(datadirs, transformation_map={'hlog': ['GFP', 'JFX549']}):
    """
    Load FCPlate objects from all FCS filesinto a dictionary.

    Parameters:
        datadirs (list of str): List of string indicating the pathes to root directories in which to search directly for FCS files.
        transformation_map (dict): Dictionary mapping transformation types to a list of channels to apply them to.

    Returns:
        plate_dict (dict of FCPlate objects): Dictionary of FCPlate objects containing all of the FCS data. Keyed by datadir basename
    """
    import os

    plate_dict = {}

    for datadir in datadirs:
        plate = load_plate(datadir, transformation_map)
        datadir_basename = os.path.basename(datadir)
        plate_dict[datadir_basename] = plate

    return plate_dict