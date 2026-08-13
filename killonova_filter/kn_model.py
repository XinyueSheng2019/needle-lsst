import numpy as np
rise_ratio      = {"u": 0.0, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.0, "y": 0.0}
rise_ratio_err  = {"u": 0.0, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.0, "y": 0.0}

decay_beta     = {"u": 2.0, "g": 1.4, "r": 0.7, "i": 0.5, "z": 0.5, "y": 0.4}
decay_beta_err = {"u": 0.2, "g": 0.2, "r": 0.2, "i": 0.2, "z": 0.2, "y": 0.2}

# peak_abs_mag =       {"u": -16, "g": -15.7, "r": -15.7, "i": -15.7, "z": -15.5, "y": -15.3}
# peak_abs_mag_err =   {"u": 1.2, "g": 1.2, "r": 1.2, "i": 1.2, "z": 1.2, "y": 1.2}

# temp_range =    {"u": 0.0, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.0, "y": 0.0}
# temp_range_err ={"u": 0.0, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.0, "y": 0.0}

photo_z = 0.1 # default redshift is 0.1, infer redshift from photometry.
rise_timescale  = 1.5 * (1 + photo_z) # by 5 mags (< 3 days, since last non-detection)
rise_timescale_err = 1.5
decay_timescale = 13.0 * (1 + photo_z) # by 5 mags
decay_timescale_err = 2.0


# start with the the same magnitude in each band then decrease correlated to ratios.
# rise_default_color_diff = {'d0': -0.1, 'd1': -0.1, 'd2': -0.1, 'd3': -0.1, 'd4': -0.1}
# fade_default_color_diff = {'d0': 0.1, 'd1': 0.1, 'd2': 0.1, 'd3': 0.1, 'd4': 0.1}


# fade more than 0.5 mag per day. 
# make them a dataframe, then compare the chi-square of the difference between the data and the model. 
# plot the multi-dimensional chi-square surface.


def fading_colour(delta_t, band1, band2, convert_to_mag=True):
    """ delta t is the time starting the peak flux"""
    delta_f1, delta_f1_err = fading_slope(delta_t, band1, convert_to_mag=convert_to_mag)
    delta_f2, delta_f2_err = fading_slope(delta_t, band2, convert_to_mag=convert_to_mag)
    return delta_f1 - delta_f2, np.sqrt(delta_f1_err**2 + delta_f2_err**2)



def fading_slope(delta_t, band, convert_to_mag = True):
    if delta_t < 0.0:
        return 0.0, 0.0
    band_flux = delta_t ** decay_beta[band]
    band_flux_err = band_flux * abs(np.log(delta_t)) * decay_beta_err[band]
    # band_flux_err = delta_t ** (decay_beta[band] + decay_beta_err[band]) - delta_t ** (decay_beta[band] - decay_beta_err[band])
    if convert_to_mag:
        return convert_flux_to_mag(band_flux, band_flux_err)
    return band_flux, band_flux_err
  

def convert_flux_to_mag(flux, flux_err):
    return -2.5 * np.log10(flux), np.round(1.0857 * flux_err / flux, 3)