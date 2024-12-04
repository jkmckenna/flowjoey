#well_functions

def count_events(well):
    """ Counts the number of events inside of a well. """
    data = well.get_data()
    count = data.shape[0]
    return count

def calculate_median_gfp(well):
    """ Calculates the median on the GFP channel. """
    data = well.get_data()
    return data['GFP'].median()