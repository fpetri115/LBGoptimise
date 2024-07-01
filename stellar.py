import stellar
import tools
import os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from sedpy import observate


def get_xsl_LSST_colours():
    """return LSST colours of XSL stars. Removes stars
    that give NAN in photometry"""
    filenames = select_approriate_xsl_files()
    bands = load_lsst_filters("./")

    photometry = []
    for file in filenames:
        spectrum = load_spectrum(file)
        mags = observate.getSED(spectrum[0]*10, spectrum[1], filterlist=bands, linear_flux=False)
        if(np.isnan(mags).all() == False):
            photometry.append(mags)

    photometry = np.vstack(np.array(photometry))
    colours = tools.calculate_colours(photometry)

    return colours

def plot_xsl_colour_diagram():
    """Plot figure 4 in https://arxiv.org/abs/2406.06850 which is done
    using SDSS filters. Get similar diagram with LSST filters which is 
    what this function does 
    """
    colours = get_xsl_LSST_colours()
    
    umg = colours[:, 0]
    gmr = colours[:, 1]
    plt.scatter(umg, gmr)

def select_approriate_xsl_files():
    """extract xsl spectra filenames
    """
    files = os.listdir("stellar_spectra_xsl/XSL_DR3_release")
    files_of_interest = []
    for filename in files:
        if("_merged.fits" in filename):
            files_of_interest.append(filename)

    return files_of_interest

def load_spectrum_fits(filename):
    """loads spectrum fits given filename
    """
    hdul = fits.open("stellar_spectra_xsl/XSL_DR3_release/"+filename)
    data_table = np.array(hdul[1].data.tolist())
    return data_table

def find_range_in_spectrum(wave_min, wave_max, wavelength, spectrum):
    """find subset of spectrum from load_spectrum()
    """
    indexes = np.where((wavelength > wave_min)&(wavelength < wave_max))[0]
    return wavelength[indexes], spectrum[indexes]

def load_spectrum(filename):
    """loads spectrum in numpy given filename
    """
    data_table = load_spectrum_fits(filename)
    wavelengths = data_table[:, 0] #nm
    flux = data_table[:, 1] #erg/s/cm2/Å
    #flux_error = data_table[:, 3]

    return wavelengths, flux

def load_lsst_filters(path):
    """gets filters for sed-py observate for calculating photometry
    """
    filters = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        filter_data = np.genfromtxt(path+'lsst_filters/total_'+band+'.dat', skip_header=7, delimiter=' ')
        filter_data[:, 0] = filter_data[:, 0]*10 #covert to angstroms
        filters.append(observate.Filter("lsst_"+band, data=(filter_data[:, 0], filter_data[:, 1])))
    
    return filters