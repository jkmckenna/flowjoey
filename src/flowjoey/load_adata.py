#load_adata

def load_adata(plate_dict, channels_to_keep=['FSC-A', 'FSC-H', 'SSC-A']):
    """
    Loads a sample sheet csv and uses one of the columns to map sample information into the AnnData object.

    Parameters:
        plate_dict (dict of FCPlate objects): The dictionary of FCPlate objects to load into the adata
        channels_to_keep (list of str): List of the channel names to keep

    Returns:
        None
    """
    import pandas as pd
    import anndata as ad

    final_adata = None
    for plate_id, plate_obj in plate_dict.items():
        for well_id, well_obj in plate_obj.data.items():
            df = well_obj.data[channels_to_keep]
            X = df.values
            adata = ad.AnnData(X, dtype=X.dtype)
            adata.obs_names = df.index.astype(str)
            adata.var_names = df.columns.astype(str)
            adata.obs['Plate'] = [f'{plate_id}'] * len(adata)
            adata.obs['Well'] = [f'{well_id}'] * len(adata)
            adata.obs['Mapping_key'] = [f'{plate_id} {well_id}'] * len(adata)
            
            if final_adata:
                if adata.shape[0] > 0:
                    final_adata = ad.concat([final_adata, adata], join='outer', index_unique=None)
            else:
                if adata.shape[0] > 0:
                    final_adata = adata.copy()

    return final_adata.copy()