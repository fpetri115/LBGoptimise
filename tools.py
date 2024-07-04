import numpy as np
from sedpy import observate

def calculate_colours(photometry):
    """given  photometry in numpy array where each column
    is a different band, calculate colours of adjacent bands
    """
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours

def load_lsst_filters(path):
    """gets filters for sed-py observate for calculating photometry
    """
    filters = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        filter_data = np.genfromtxt(path+'lsst_filters/total_'+band+'.dat', skip_header=7, delimiter=' ')
        filter_data[:, 0] = filter_data[:, 0]*10 #covert to angstroms
        filters.append(observate.Filter("lsst_"+band, data=(filter_data[:, 0], filter_data[:, 1])))
    
    return filters