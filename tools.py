import numpy as np

def calculate_colours(photometry):
    """given  photometry in numpy array where each column
    is a different band, calculate colours of adjacent bands
    """
    photo1 = photometry[:,:-1]
    photo2 = photometry[:,1:]
    colours = photo1 - photo2

    return colours