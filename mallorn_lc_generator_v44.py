import sncosmo
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gp
import pandas as pd
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import random
import os
import time 
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import WMAP9
import rubin_sim.maf as maf  
import rubin_scheduler.utils as rsUtils
from rubin_sim.data import get_baseline 
from astropy.coordinates import SkyCoord 
from astropy.cosmology import z_at_value 
from dustmaps.sfd import SFDQuery
import dustmaps.sfd 
import extinction 
from extinction import fitzpatrick99
from dustmaps.config import config
from eztao.carma import DRW_term
from eztao.ts import gpSimRand
from eztao.ts import drw_fit 
import scipy.optimize
import unicodedata
import re
import warnings
from astropy.utils.exceptions import AstropyWarning
import celerite

from bsline_crctn import inverse_variance_weighting
from bsline_crctn import baseline_correction_4filter
# ^ From Vysakh

#####   Load data   ##### 

split_num = '01'

overall_start_time = time.time()
#Defining start time - used to find total start time of code

run_time_list = []
lsst_peaks_list = []
mse_list = []
redshift_list = []
#Defining lists that will be saved to for each lightcurve created

base_dir = f'/Users/dylanmagill/python/ZTF_cls_data_ext_test/folder_{split_num}/'
# ^ Change this to whatever your base directory is
#sub_dirs = ['Not-AGN']
all_files = []
path = base_dir
for root, dirs, files in os.walk(path):
    for file_name in files:
        if file_name.endswith('_clean_data.txt'):
            all_files.append(os.path.join(root, file_name))
#print(f"All Files: {all_files}")
#Getting list of all lightcurve files

cols = ['time_jd', 'flux', 'flux_unc', 'zeropt', 'filter']
#Assigning names for each column of the data table

Ia_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF18aadzfso/ZTF18aadzfso_clean_data.txt'
#Path for a SN Ia
Ia_91T_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF21aaahliy/ZTF21aaahliy_clean_data.txt'
#Path for a SN Ia-91T
Ia_91bg_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF22aakejxf/ZTF22aakejxf_clean_data.txt'
#Path for a SN Ia-91bg
Ib_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF20aalcyih/ZTF20aalcyih_clean_data.txt'
#Path for a SN Ib 
Ibc_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF22abecedy/ZTF22abecedy_clean_data.txt'
#Path for a SN Ib/c
#path = N/A
#Path for a SN Ib-pec
Ic_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF19acjtpqd/ZTF19acjtpqd_clean_data.txt'
#Path for a SN Ic
Ic_BL_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF20aaurexl/ZTF20aaurexl_clean_data.txt'
#Path for a SN Ic-BL
II_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF20aapycrh/ZTF20aapycrh_clean_data.txt'
#Path for a SN II
IIP_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF19abqyouo/ZTF19abqyouo_clean_data.txt'
#Path for a SN IIP
IIb_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF21aaxxmvs/ZTF21aaxxmvs_clean_data.txt'
#Path for a SN IIb
IIn_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF20aaurfwa/ZTF20aaurfwa_clean_data.txt'
#Path for a SN IIn
SLSNI_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF19acbonaa/ZTF19acbonaa_clean_data.txt'
#Path for a SLSN-I
SLSNII_path = '/Users/dylanmagill/python/ZTF_cls_data_ext_test/folder_06/ZTF22aabomyi/ZTF22aabomyi_clean_data.txt'
#Path for a SLSN-II
TDE_path = '/Users/dylanmagill/python/ZTF_cls_data/Not-AGN/ZTF21abcgnqn/ZTF21abcgnqn_clean_data.txt'
#Path for a TDE
AGN_path = '/Users/dylanmagill/python/ZTF_cls_data/AGN/ZTF19aayiili/ZTF19aayiili_clean_data.txt'
#Path fo an AGN
AGN2_path = '/Users/dylanmagill/Documents/ZTF_Data_24/19/ZTF19aatyahe/ZTF19aatyahe_clean_data.txt'
AGN3_path = '/Users/dylanmagill/Documents/ZTF_Data_24/19/ZTF19aavrrty/ZTF19aavrrty_clean_data.txt'

test_path = '/Users/dylanmagill/Documents/ZTF_Data_24/20/ZTF20acitpfz/ZTF20acitpfz_clean_data.txt'
amon_curunir_agor = '/Users/dylanmagill/Downloads/ZTF_cls_data_ext_test/folder_01/ZTF21aaixlfe/ZTF21aaixlfe_clean_data.txt'

path = TDE_path
#Choose what type of object you want - for testing purposes

express_tag = False
#Adding express tag for faster computation if testing something

bulk_tag = False; bulk_num = len(all_files)
#Adding bulk tag to allow for creation of multiple lightcurves

sub_run_tag = True; sub_num_lightcurves = 10
#Adding sub run tag to allow for creation of multiple lightcurves from a single GP fit

DDF_tag = False
#Setting DDF tag. True = DDF, False = WFD.

#output_tag = 'real'
output_tag = 'test'
#Defining output tag to determine what files and graphs are produced
#Real produces only the necessary files. Test produces more graphs to allow for performance evaluation

if bulk_tag == True:
    num_lightcurves = bulk_num
    selected_files = all_files
#Randomly selecting a chosen number of lightcurve files
else:
    num_lightcurves = 1
    selected_files = []
    selected_files.append(path)
#Choosing selected path for non-bulk mode

if sub_run_tag == False:
    sub_num_lightcurves = 1
#If false only make one lightcurve per GP fit

#output_path = '/Users/dylanmagill/python/lsst_sim_phot_outputs/'
#Defining path for output directory

master_output_path = f'/Users/dylanmagill/python/MALLORN/split_{split_num}/'
os.makedirs(master_output_path, exist_ok=True)
if output_tag == 'test':
    sim_output_path = f'/Users/dylanmagill/python/MALLORN/split_{split_num}/sim_outputs/'
    os.makedirs(sim_output_path, exist_ok=True)
#Defining paths for output files and making relevant folders

def ab_to_uJy (magAB):
    flux_Jy = 10**(23 - (magAB + 48.6)/2.5)
    flux_mJy = flux_Jy*1000000
    return flux_mJy
#Defining a function to convert from mag to flux

def uJy_to_ab (mJy):
    flux_Jy = mJy/1000000
    ab = 2.5*(23 - np.log10(flux_Jy))-48.6
    return ab
#Defining a function to convert from flux to mag

config['data_dir'] = '/Users/dylanmagill/python/dustmaps'
#Importing necessary packages and setting path for dustmap

dustmaps.sfd.fetch()
#Loading in the dustmap - only needed once, sometimes crashes if the Harvard servers aren't running, hence commented out when not needed

def jurassic_park (flux, eff_wl):
    A_lambda = fitzpatrick99(eff_wl, ebv * 3.1) #3.1 = Standard Milky Way value
    flux_ext = flux * 10**((A_lambda)/2.5)
    return flux_ext
#Defining function to de-extinct ZTF lightcurves

filter_colours = {'u': '#6A5ACD', 'g': '#2ca02c', 'r': '#d62728', 'i': '#ff7f0e', 'z': '#8c564b', 'y': '#1b1b1b'}
#Defining filter colours

accepted_spec_types = ['SN Ia', 'SN Ia-91T-like', 'SN Ia-91bg-like', 'SN Iax[02cx-like]', 'SN Ia-02es-like', 'SN Ia-pec', 'SN Ib', 'SN Ib/c', 'SN Ib-pec',
                       'SN Ic', 'SN Ic-BL', 'SN II', 'SN IIP', 'SN IIb', 'SN IIn', 'SLSN-I', 'SLSN-II', 'TDE', 'AGN']
#Defining list of accepted spectral types which the code can handle

Ia_num = 0; Ia91T_num = 0; Ia91bg_num = 0; Ia02cx_num = 0; Ia02es_num = 0; Iapec_num = 0; Ib_num = 0; Ibc_num = 0; Ibpec_num = 0
Ic_num = 0; IcBL_num = 0; II_num = 0; IIb_num = 0; IIn_num = 0; SLSNI_num = 0; SLSNII_num = 0; TDE_num = 0; AGN_num = 0

imrad = '/Users/dylanmagill/python/sindarin.csv'
thafn = ['Sindarin', 'English']
ist = pd.read_csv(imrad, sep=",", skiprows=0,names=thafn, usecols=thafn)
peth_lest = ist.iloc[:,0].tolist()
#Defining list of Sindarin words

def sanitize_filename(word):
    nfkd_form = unicodedata.normalize('NFKD', word)
    ascii_only = nfkd_form.encode('ASCII', 'ignore').decode('ASCII')
    clean = re.sub(r'[^a-zA-Z0-9-]', '', ascii_only)
    return clean
#Function to replace letters with accents, ensuring the filename doesn't cause any issues

def generate_elvish_name():
    chosen_rows = ist.sample(n=3)
    #Choose two rows at random

    peth_min = chosen_rows.iloc[0]['Sindarin']
    peth_tad = chosen_rows.iloc[1]['Sindarin']
    peth_nel = chosen_rows.iloc[2]['Sindarin']
    eng_min = chosen_rows.iloc[0]['English'] 
    eng_tad = chosen_rows.iloc[1]['English']
    eng_nel = chosen_rows.iloc[2]['English']
    #Choosing the random words and getting their corresponding English meaning

    peth_min_puig = sanitize_filename(peth_min)
    peth_tad_puig = sanitize_filename(peth_tad)
    peth_nel_puig = sanitize_filename(peth_nel)
    narthan = peth_min_puig + '_' + peth_tad_puig + '_' + peth_nel_puig
    #Removing any accents

    english_name = eng_min + ' + ' + eng_tad + ' + ' + eng_nel
    #Getting the corresponding English meaning

    print(f"Sindarin: {peth_min} + {peth_tad} + {peth_nel}")
    print(f"English: {english_name}")
    print(f"Filename: {narthan}")

    return narthan, english_name

master_log_filename = master_output_path + "master_log.csv"
log_columns = ["Sindarin Name", "English Translation", "ZTF Name", "SpecType", "Z (ZTF)", "Z (LSST)", "Peak Time", "LSST r Peak", "GP Attempts", "MSE", "Model", "Sim RA", "Sim Dec", "Fudge", "Cadence Attempts", "5 Sigma Points", "EBV", "DDF Tag", "Output Location", "Run Time"]
if not os.path.exists(master_log_filename):
    pd.DataFrame(columns=log_columns).to_csv(master_log_filename, index=False)
#Creating master log file

fail_log_filename = master_output_path + "fail_log.csv"
fail_log_cols = ["ZTF_Name", "SpecType", "Fail Reason"]
if not os.path.exists(fail_log_filename):
    pd.DataFrame(columns=fail_log_cols).to_csv(fail_log_filename, index=False)

lc_cols = ["Object_ID", "Time (MJD)", "Flux", "Flux_err", "Filter"]
#Defining output columns

train_full_lightcurve_filename = master_output_path + "train_full_lightcurves_premix.csv"
if not os.path.exists(train_full_lightcurve_filename):
    pd.DataFrame(columns=lc_cols).to_csv(train_full_lightcurve_filename, index=False)
train_short_lightcurve_filename = master_output_path + "train_short_lightcurves_premix.csv"
if not os.path.exists(train_short_lightcurve_filename):
    pd.DataFrame(columns=lc_cols).to_csv(train_short_lightcurve_filename, index=False)
#Defining training set lightcurve files

test_pub_full_lightcurve_filename = master_output_path + "test_pub_full_lightcurves_premix.csv"
if not os.path.exists(test_pub_full_lightcurve_filename):
    pd.DataFrame(columns=lc_cols).to_csv(test_pub_full_lightcurve_filename, index=False)
test_pub_short_lightcurve_filename = master_output_path + "test_pub_short_lightcurves_premix.csv"
if not os.path.exists(test_pub_short_lightcurve_filename):
    pd.DataFrame(columns=lc_cols).to_csv(test_pub_short_lightcurve_filename, index=False)
#Defining public testing set lightcurve files

test_priv_full_lightcurve_filename = master_output_path + "test_priv_full_lightcurves_premix.csv"
if not os.path.exists(test_priv_full_lightcurve_filename):
    pd.DataFrame(columns=lc_cols).to_csv(test_priv_full_lightcurve_filename, index=False)
test_priv_short_lightcurve_filename = master_output_path + "test_priv_short_lightcurves_premix.csv"
if not os.path.exists(test_priv_short_lightcurve_filename):
    pd.DataFrame(columns=lc_cols).to_csv(test_priv_short_lightcurve_filename, index=False)
#Defining private testing set lightcurve files

obj_log_cols = ["Object_ID", "Z", "Z_err", "EBV", "SpecType", "English Translation"]

train_log_filename = master_output_path + "train_log_premix.csv"
if not os.path.exists(train_log_filename):
    pd.DataFrame(columns=obj_log_cols).to_csv(train_log_filename, index=False)
test_pub_log_filename = master_output_path + "test_pub_log_premix.csv"
if not os.path.exists(test_pub_log_filename):
    pd.DataFrame(columns=obj_log_cols).to_csv(test_pub_log_filename, index=False)
test_priv_log_filename = master_output_path + "test_priv_log_premix.csv"
if not os.path.exists(test_priv_log_filename):
    pd.DataFrame(columns=obj_log_cols).to_csv(test_priv_log_filename, index=False)


for run in range(num_lightcurves):
    run_start_time = time.time()
    #Defining start time for lightcurve creation

    path = selected_files[run]
    #Selecting object for lightcurve

    df = pd.read_csv(path, sep=r"\s+", skiprows=3,names=cols, usecols=cols)
    #Reading in the text file for the clean dataset

    flux = df.iloc[:,1]
    times = df.iloc[:,0]
    #Defining fluxes and times

    r_flux = []; r_time = []; r_error = []
    g_flux = []; g_time = []; g_error = []
    #Creating lists to save values to for each respective filter

    r_mask = df['filter'] == 'ZTF_r'
    r_time = np.array(df.loc[r_mask, 'time_jd'])
    r_flux = np.array(df.loc[r_mask, 'flux'])
    r_error = np.array(df.loc[r_mask, 'flux_unc'])
    r_zpt_list = df.loc[r_mask, 'zeropt'].tolist()
    g_mask = df['filter'] == 'ZTF_g'
    g_time = np.array(df.loc[g_mask, 'time_jd'])
    g_flux = np.array(df.loc[g_mask, 'flux'])
    g_error = np.array(df.loc[g_mask, 'flux_unc'])
    g_zpt_list = df.loc[g_mask, 'zeropt'].tolist()
    #Saving respective values according to their filter

    no_data_flag_r = False
    no_data_flag_g = False

    if len(r_flux) < 0:
        no_data_flag_r = True

    try:
        if len(r_flux) > 0:
            r_zpt = np.mean(r_zpt_list)
            r_flux = r_flux * ab_to_uJy(r_zpt)
            r_error = r_error * ab_to_uJy(r_zpt)
            #Determining zeropoint values and converting flux from DN to uJy

            r_peak = max(r_flux)
            r_peak_time = r_time[np.argmax(r_flux)]
            #Determining peak and peak time

            r_flux, r_error, r_crctn_data, bl_flag = baseline_correction_4filter(r_flux, r_error, r_time, goback=100, peak_time=r_peak_time)
            #Applying baseline correction from Vysakh's function
        else:
            print("Skipping r-band — no data.")
            r_peak = None
            r_peak_time = None
            r_crctn_data = None
            no_data_flag_r = True
    except Exception as e:
        print(f"Error in r-band handling: {e}")
        r_peak = None
        r_peak_time = None
        r_crctn_data = None
        no_data_flag_r = True
    #Wrapped in try statement to prevent crashing if no r band data present

    if len(g_flux) < 20:
        no_data_flag_g = True

    try:
        if len(g_flux) > 0:
            g_zpt = np.mean(g_zpt_list)
            g_flux = g_flux * ab_to_uJy(g_zpt)
            g_error = g_error * ab_to_uJy(g_zpt)
            #Determining zeropoint values and converting flux from DN to uJy

            g_peak = max(g_flux)
            g_peak_time = g_time[np.argmax(g_flux)]
            #Determining peak and peak time

            g_flux, g_error, g_crctn_data, bl_flag = baseline_correction_4filter(g_flux, g_error, g_time, goback=100, peak_time=r_peak_time)
            #Applying baseline correction from Vysakh's function
        else:
            print("Skipping g-band — no data.")
            g_peak = None
            g_peak_time = None
            g_crctn_data = None
            no_data_flag_g = True
    except Exception as e:
        print(f"Error in g-band handling: {e}")
        g_peak = None
        g_peak_time = None
        g_crctn_data = None
        no_data_flag_g = True
    #Wrapped in try statement to prevent crashing if no g band data present

    if no_data_flag_r and no_data_flag_g:
        fail_reason = "No data"
        fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                                "SpecType": SpecType,
                                "Fail Reason": fail_reason}])
        fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
        continue

    #####   Determining redshift & object type   #####

    object_name = [path.split('/')[-2]]
    print(object_name)
    #Getting the object name

    dave_path = '/Users/dylanmagill/python/ztf_cls_ext.csv'
    cols2 = ['TNSName','discoveryName','raDeg','decDeg','discDate','discMag','discMagFilter','discSurvey','objectUrl','obsdate',
         'reportAddedDate','specType','survey','telescope','transRedshift','hostName','hostRedshift','reportingSurvey','TNSuser']
    dave_df = pd.read_csv(dave_path, skiprows=0,names=cols2, usecols=cols2)
    dave_df = dave_df.dropna(subset=['specType'])
    #Loading in classification file

    index_list = dave_df.query("discoveryName == @object_name[0]").index.tolist()
    #Query list to find index of matching name

    if not index_list:
        print("No matching object found.")
        fail_reason = "No matching object found"
        fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                                "SpecType": SpecType,
                                "Fail Reason": fail_reason}])
        fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
        continue
        #Print statement in case of error

    redshift = dave_df.loc[index_list[0], 'transRedshift']
    SpecType = dave_df.loc[index_list[0], 'specType']
    obj_RA = int(float(dave_df.loc[index_list[0], 'raDeg']))
    obj_Dec = int(float(dave_df.loc[index_list[0], 'decDeg']))
    print(f"Redshift: {redshift}")
    print(f"SpecType: {SpecType}")
    print(f"RA (deg): {obj_RA}, Dec (deg): {obj_Dec}")
    if SpecType == 'AGN':
        AGN_tag = True
    else:
        AGN_tag = False
    #Get corresponding redshift, type and RA/Dec for that object
    if SpecType not in accepted_spec_types:
        print(f'Object {object_name} not accepted type ({SpecType}). Skipping to next object...')
        fail_reason = 'Object not accepted type'
        fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                                "SpecType": SpecType,
                                "Fail Reason": fail_reason}])
        fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
        continue
    #Skips to next object if not an accepted spectral type covered by SNCosmo

    if output_tag == 'test':
        obj_output_path = sim_output_path + '/' + object_name[0]
        #obj_output_path = f'/Users/dylanmagill/python/lsst_sim_phot_outputs/{object_name}'
        os.makedirs(obj_output_path, exist_ok=True)
        #Making folder to store outputs

    ZTF_coords = SkyCoord(obj_RA,obj_Dec, unit='deg')
    SFD = SFDQuery()
    ebv = SFD(ZTF_coords) * 0.86 #Schafly & Finkbeiner 2011
    #Calculating ebv value using SFD query

    g_eff_wl_ztf = np.array([4746]); r_eff_wl_ztf = np.array([6366])
    #Effective wavelength for each band - sourced from SVO Filter Profile Service for Palomar-ZTF

    pre_ext_g_flux_ztf = g_flux; pre_ext_r_flux_ztf = r_flux
    #Saving old values to check de-extinction is working

    g_flux = jurassic_park(g_flux,g_eff_wl_ztf) ;r_flux = jurassic_park(r_flux,r_eff_wl_ztf)
    #De-extincting ZTF lightcurves

    #####   Determine which band has the most data   #####

    r_size = len(r_flux); g_size = len(g_flux)
    if r_size > g_size:
        main_band_flux = r_flux
        main_band_time = r_time
        main_band_error = r_error
        main_band_tag = 'r'
        print("R Band Selected")
        main_band_peak = max(main_band_flux)
        main_band_peak_time = main_band_time[np.argmax(main_band_flux)]
        #Defining the peak values and times
        main_band_time = np.array(main_band_time) - main_band_peak_time
        init_main_band_peak_time = main_band_peak_time
        main_band_peak_time = 0
        #Setting time relative to peak date
        rise_start_mask = main_band_flux > 5 * main_band_error
        if np.any(rise_start_mask):
            main_band_rise_start = main_band_time[np.where(rise_start_mask)[0][0]]
        else:
            main_band_rise_start = (main_band_peak_time - 10)
        print(main_band_rise_start)

        #Determining time of explosion for the Nugent models
    else:
        main_band_flux = g_flux
        main_band_time = g_time
        main_band_error = g_error
        main_band_tag = 'g'
        print("G Band Selected")
        main_band_peak = max(main_band_flux)
        main_band_peak_time = main_band_time[np.argmax(main_band_flux)]
        #Defining the peak value and time
        main_band_time = np.array(main_band_time) - main_band_peak_time
        init_main_band_peak_time = main_band_peak_time
        main_band_peak_time = 0
        #Setting time relative to peak date
        rise_start_mask = main_band_flux > 5 * main_band_error
        if np.any(rise_start_mask):
            main_band_rise_start = main_band_time[np.where(rise_start_mask)[0][0]]
        else:
            main_band_rise_start = (main_band_peak_time - 10)
        print(main_band_rise_start)
        #Determining time of explosion for the Nugent models

    #####   Generating GP Fit   #####

        #####   MSE Test to Determine Accuracy of Fit   #####

    def reduce_array_with_distance(x, y, errs, *, retain_fraction=0.8, min_distance=5.0):
            x, y, errs = np.array(x), np.array(y), np.array(errs)
            indices, retain_size = np.arange(len(x)), int(len(x) * retain_fraction)
            removed_indices = []
            
            while len(indices) > retain_size:
                idx_to_check = np.random.choice(indices)
                distances = np.abs(x[indices] - x[idx_to_check])
                if np.any((distances < min_distance) & (distances > 0)):
                    removed_indices.append(idx_to_check)
                    indices = indices[indices != idx_to_check]
            
            return x[indices], y[indices], errs[indices], x[removed_indices], y[removed_indices]
    #Defining the Reduction function

    def GP_MSE(main_band_flux,main_band_time,main_band_error):

        mb_normalised = (main_band_flux - np.mean(main_band_flux)) / np.std(main_band_flux)
        mb_error_normalised = np.abs((main_band_error - np.mean(main_band_error)) / np.std(main_band_error))
        #Normalising the flux and errors

        mbx_reduced, mby_reduced, errs_mb_reduced, mbx_removed, mby_removed = reduce_array_with_distance(main_band_time, mb_normalised, mb_error_normalised)
        #Removing values according to reduction function

        mb_ls = round(len(mby_reduced) / 40)
        kernel_mb_mse = C(1.0) * Matern(length_scale=mb_ls, length_scale_bounds=(1e-3, 1e3), nu=3/2) + WhiteKernel(noise_level=0.05)
        model_mb_mse = gp.GaussianProcessRegressor(kernel=kernel_mb_mse, n_restarts_optimizer=20, alpha=errs_mb_reduced**2, normalize_y=False)
        #Defining model on reduced value
        
        model_mb_mse.fit(np.array([mbx_reduced]).T, mby_reduced)
        y_mb_pred, _ = model_mb_mse.predict(np.array([mbx_removed]).T, return_std=True)
        #Producing values for the missing points from the reduced GP fit
        
        mse_mb = mean_squared_error(mby_removed, y_mb_pred)
        mse_mb = round(mse_mb, 4)
        #Calculating MSE value

        return mse_mb
    
    mse_check = 0.3

    def gp_lc_fit (flux, time, error, peak, max_gp_retries = 4, threshold_ratio = 0.9):
        mse_success_tag = False
        #Setting to false by default

        ls_gp_base = round(len(flux)/40)
        #Setting length scale

        errs_gp = np.array(error)
        #Adjusting format of errors to work in gp fit

        best_gp_peak_ratio = 0
        #Setting placeholder best peak ratio value

        best_ls = ls_gp_base
        best_errs = errs_gp

        for attempt in range(max_gp_retries):

            # if AGN_tag == True:

            #     if max(time) - min(time) < 30:
            #         print("Too short a lightcurve for DRW fit.")
            #         return None, None, attempt, 'NaN', False

            #     try:
            #         agn_best_fit = drw_fit(time, flux, error)
            #         print(f'Best-fit DRW parameters: {agn_best_fit}')

            #     except AssertionError as e:
            #         print(f"DRW fit failed for {object_name[0]}: {e}")
            #         fail_reason = 'Invalid DRW parameter bounds'
            #         fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
            #                     "SpecType": SpecType,
            #                     "Fail Reason": fail_reason}])
            #         fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
            #         return None, None, attempt, 'NaN', False

            #     #agn_best_fit = drw_fit(time, flux, error)

            #     amp, tau = agn_best_fit

            #     agn_duration = max(time) - min(time)

            #     DRW_kernel = DRW_term(np.log(amp), np.log(tau))
            #     #gp_timescale, gp_flux, yerr = gpSimRand(DRW_kernel, 100, agn_duration, 1000)

            #     gp_timescale = time
            #     gp_flux = gpSimRand(DRW_kernel, time, error)   # (depending on your gpSimRand implementation)

            #     gp_flux = gp_flux + np.mean(flux)

            #     plt.errorbar(time, flux, yerr=error, fmt='.', color='red', label='observed')
            #     plt.plot(time, gp_flux, color='orange', label='DRW fit')
            #     plt.legend()
            #     plt.show()

            #     mse_mb = 'NaN'
            #     mse_success_tag = True
            
            #     return gp_flux, gp_timescale, attempt, mse_mb, mse_success_tag
            #Runs AGN damped random walk code from Xinyue for any AGN

            if AGN_tag:
                if max(time) - min(time) < 30:
                    print("Too short a lightcurve for DRW fit.")
                    return None, None, attempt, 'NaN', False

                try:
                    amp, tau = drw_fit(time, flux, error)
                    print(f'Best-fit DRW parameters: amp={amp}, tau={tau}')
                except AssertionError as e:
                    print(f"DRW fit failed for {object_name[0]}: {e}")
                    fail_reason = 'Invalid DRW parameter bounds'
                    fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                                                    "SpecType": SpecType,
                                                    "Fail Reason": fail_reason}])
                    fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
                    return None, None, attempt, 'NaN', False

                #Build DRW kernel and GP (celerite)
                kernel = DRW_term(np.log(amp), np.log(tau))
                gp_agn = celerite.GP(kernel, mean=np.mean(flux))
                gp_agn.compute(time, error)

                #Predict DRW model on dense grid
                gp_timescale = np.linspace(min(time), max(time), 1000)
                gp_flux, agn_var = gp_agn.predict(flux, gp_timescale, return_var=True)

                #Plot observed vs fit
                #plt.errorbar(time, flux, yerr=error, fmt='.', color='red', label='observed')
                #plt.plot(gp_timescale, gp_flux, color='orange', label='DRW fit')
                #plt.legend()
                #plt.show()

                mse_mb = 'NaN'

                mse_success_tag = True

                return gp_flux, gp_timescale, attempt, mse_mb, mse_success_tag


            ls_guess = ls_gp_base * np.random.uniform(0.7,1.2)
            noise_guess = np.random.uniform(0.01,0.1)
            amplitude_guess = peak * np.random.uniform(0.8, 1.2)
            #Creating guess values to allow for variation between runs

            if (attempt + 1) == max_gp_retries:
                ls_gp_base = best_ls
                errs_gp = best_errs
            #Chooses best parameters for final attempt

            kernel = C(amplitude_guess, (peak*0.01, peak*100)) * Matern(length_scale=ls_guess, length_scale_bounds=(1e-3, 1e3), nu=3/2) + WhiteKernel(noise_level=noise_guess, noise_level_bounds=(1e-6,1e3))
            model_gp = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=(errs_gp * 0.9)**2, normalize_y=False)
            #Creating kernel and model for main band - reducing error values in order to capture peak

            model_gp.fit(np.array([time]).T,flux)
            params_gp = model_gp.kernel_.get_params()
            #Fitting model for main band

            gp_timescale = np.linspace(min(time), max(time), 1000)
            gp_flux, std_gp = model_gp.predict(np.array([gp_timescale]).T, return_std=True)
            #Creating model values for main band

            gp_max_flux = np.max(gp_flux)
            peak_ratio = gp_max_flux/peak
            print('GP Peak / Obs Peak = ',peak_ratio)
            #Determining ratio between GP peak and observed peak

            mse_mb = GP_MSE(flux,time,error)

            if express_tag == True:
                print(f'GP Model Created (*Express Mode* - peak captured')
                mse_success_tag = True
                break
            #Skipping if express tag is true

            if gp_max_flux >= threshold_ratio * peak and mse_mb < mse_check:
                print(f'GP Model Created (attempt {attempt + 1}) - peak captured')
                mse_success_tag = True
                break
            #Breaking if GP model successfully captures the peak
            else:
                if peak_ratio > best_gp_peak_ratio:
                    best_ls = ls_guess
                    best_errs = errs_gp
                #Saving values if better than previous run. Best values used for final run.
                ls_gp_base *= 0.5
                errs_gp *= 0.7
                print(f'GP fit attempt {attempt + 1} failed to reach peak. Retrying fit...')
                #Adjusting values for next attempt at GP fit

        else:
            print(f'Warning: GP fit failed to capture peak flux after {max_gp_retries} attempts')
            print('GP Peak / Obs Peak = ',peak_ratio)
            if mse_mb < mse_check:
                mse_success_tag = True
        #Determining if GP fit has effectively fit the peak. If yes, passes on. If no, alters values and reruns GP fit until it does.

        return gp_flux, gp_timescale, attempt, mse_mb, mse_success_tag
    #Defining function for GP fit to a lightcurve

    main_band_flux_gp, gp_timescale, main_band_attempt, mse_mb, mse_success_tag = gp_lc_fit (main_band_flux, main_band_time, main_band_error, main_band_peak)
    #Carrying out GP fit for the main band

    #print(gp_timescale)

    if mse_success_tag == False:
        if AGN_tag == True:
            print(f"DRW fit failed for {object_name[0]} due to insufficient data")
            fail_reason = 'Too short a lightcurve for DRW fit'
            fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                        "SpecType": SpecType,
                        "Fail Reason": fail_reason}])
            fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
        if AGN_tag == False:
            print(f"GP fit failed for {object_name[0]} due to poor MSE value [{mse_mb}] — skipping object.")
            fail_reason = 'Poor MSE value'
            fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                        "SpecType": SpecType,
                        "Fail Reason": fail_reason}])
            fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
        continue

    print('MSE = ', mse_mb)
    mse_list.append(mse_mb)

    #####   Creating Blackbody Model   #####

    def planck_lambda(wavelength, temperature, phase):
        # Constants
        h = 6.62607015e-27  # Planck's constant in erg*s
        c = 2.998e10  # Speed of light in cm/s
        k_B = 1.380649e-16  # Boltzmann constant in erg/K

        lam_cm = wavelength * 1e-8
        # Convert wavelength from Å to cm

        factor1 = (2.0 * h * c**2) / (lam_cm**5)
        exponent = (h * c) / (lam_cm * k_B * temperature)
        intensity = factor1 / (np.exp(exponent) - 1.0)
        # Planck's Law

        intensity2 = []
        for i in phase:
            intensity2.append(intensity)
        # Creating array of intensities for phase component

        return np.array(intensity2)  # erg / (s cm² Å)

    class Blackbody(sncosmo.Source):
        _param_names = ['amplitude', 'temperature']
        param_names_latex = ['A', 'T']   # used in plotting display

        def __init__(self, name='Blackbody', version='1.0'):
            self.name = name
            self.version = version
            self._phase_min = 0.0
            self._phase_max = 100.0
            self._wave_min = 100.0
            self._wave_max = 30000.0
            self._wave = np.linspace(self._wave_min, self._wave_max, 1000)  # 1D wavelength grid
            self._phase = np.linspace(self._phase_min, self._phase_max, 100)  # 1D phase grid
            self._restmin = self._wave_min
            self._restmax = self._wave_max
            self._parameters = np.array([1.0, 6000.0])  # default amplitude = 1, temperature = 6000K

        def _flux(self, phase, wave):
            # wave should be 1D array
            amp, temp = self._parameters
            flux = amp * planck_lambda(wave, temp, phase)  # Calculate the flux using Planck's law
            return flux
        
    #Need to subtract dT_dt from T for first 100 days after peak

    def dT_dt_planck_lambda(wavelength, temperature, dT_dt, phase):
        # Constants
        h = 6.62607015e-27  # Planck's constant in erg*s
        c = 2.998e10  # Speed of light in cm/s
        k_B = 1.380649e-16  # Boltzmann constant in erg/K

        lam_cm = wavelength * 1e-8
        # Convert wavelength from Å to cm

        #alt_temperature = temperature - (dT_dt * phase)

        #factor1 = (2.0 * h * c**2) / (lam_cm**5)
        #exponent = (h * c) / (lam_cm * k_B * alt_temperature)
        #intensity = factor1 / (np.exp(exponent) - 1.0)
        # Planck's Law

        intensity2 = []
        for ph in phase:
            effective_phase = np.clip(ph,0,100)
            #Defining effective phase as peak time to 100 days after peak

            cap_temperature = 40000
            floor_temperature = 8000

            alt_temperature = temperature - (dT_dt * effective_phase)
            alt_temperature = np.clip(alt_temperature, floor_temperature, cap_temperature)
            #Altering temperature according to time after peak

            factor1 = (2.0 * h * c**2) / (lam_cm**5)
            exponent = (h * c) / (lam_cm * k_B * alt_temperature)
            intensity = factor1 / (np.exp(exponent) - 1.0)
            # Planck's Law
            
            intensity2.append(intensity)
        # Creating array of intensities for phase component

        return np.array(intensity2)  # erg / (s cm² Å)


    class dT_dt_Blackbody(sncosmo.Source):
        _param_names = ['amplitude', 'temperature', 'dT_dt']
        param_names_latex = ['A', 'T', 'dT_dt']   # used in plotting display

        def __init__(self, name='dT_dt_Blackbody', version='1.0'):
            self.name = name
            self.version = version
            self._phase_min = 0.0
            self._phase_max = 100.0
            self._wave_min = 100.0
            self._wave_max = 30000.0
            self._wave = np.linspace(self._wave_min, self._wave_max, 1000)  # 1D wavelength grid
            self._phase = np.linspace(self._phase_min, self._phase_max, 100)  # 1D phase grid
            self._restmin = self._wave_min
            self._restmax = self._wave_max
            self._parameters = np.array([1.0, 6000.0, 10])  # default amplitude = 1, temperature = 6000K, dT/dt = 10K/day

        def _flux(self, phase, wave):
            # wave should be 1D array
            amp, temp, dT_dt = self._parameters
            flux = amp * dT_dt_planck_lambda(wave, temp, dT_dt, phase)  # Calculate the flux using Planck's law
            return flux
        
    def dT_dt_planck_lambda_SLSN(wavelength, temperature, dT_dt, phase):
        # Constants
        h = 6.62607015e-27  # Planck's constant in erg*s
        c = 2.998e10  # Speed of light in cm/s
        k_B = 1.380649e-16  # Boltzmann constant in erg/K

        lam_cm = wavelength * 1e-8
        # Convert wavelength from Å to cm

        lam_crit_cm = 3000 * 1e-8
        #Defining critical wavelength of 3000Å and converting to cm

        intensity2 = []
        for ph in phase:
            effective_phase = np.clip(ph,0,70)
            #Defining effective phase as peak time to 50 days after peak

            alt_temperature = temperature - (dT_dt * effective_phase)
            #Altering temperature according to time after peak

            factor1 = (2.0 * h * c**2) / (lam_cm**5)
            #factor2 = lam_cm/lam_crit_cm if lam_cm < lam_crit_cm else 1
            factor2 = np.where(lam_cm < lam_crit_cm, lam_cm / lam_crit_cm, 1)
            exponent = (h * c) / (lam_cm * k_B * alt_temperature)
            intensity = factor1 * factor2 / (np.exp(exponent) - 1.0)
            # Planck's Law
            
            intensity2.append(intensity)
        # Creating array of intensities for phase component

        return np.array(intensity2)  # erg / (s cm² Å)


    class dT_dt_Blackbody_SLSN(sncosmo.Source):
        _param_names = ['amplitude', 'temperature', 'dT_dt']
        param_names_latex = ['A', 'T', 'dT_dt']   # used in plotting display

        def __init__(self, name='dT_dt_Blackbody_SLSN', version='1.0'):
            self.name = name
            self.version = version
            self._phase_min = 0.0
            self._phase_max = 100.0
            self._wave_min = 100.0
            self._wave_max = 30000.0
            self._wave = np.linspace(self._wave_min, self._wave_max, 1000)  # 1D wavelength grid
            self._phase = np.linspace(self._phase_min, self._phase_max, 100)  # 1D phase grid
            self._restmin = self._wave_min
            self._restmax = self._wave_max
            self._parameters = np.array([1.0, 6000.0, 10])  # default amplitude = 1, temperature = 6000K, dT/dt = 10K/day

        def _flux(self, phase, wave):
            # wave should be 1D array
            amp, temp, dT_dt = self._parameters
            flux = amp * dT_dt_planck_lambda_SLSN(wave, temp, dT_dt, phase)  # Calculate the flux using Planck's law
            return flux

    pre_sub_run_time = time.time() - run_start_time
    #Defining time taken to get to this point

    nothrim = set()
    #Creating set to list Sindarin names

    for sub_num in range(sub_num_lightcurves):
    #Loop to create multiple lightcurves for the one GP fit

        print(f'subnum = {sub_num}')

        sub_run_start_time = time.time()
        #Taking time at start of run

        if sub_num < 3:
            training_set_tag = True
            pub_test_tag = False
            priv_test_tag = False
        else:
            training_set_tag = False
            if random.random() < 0.5:
                pub_test_tag = True
                priv_test_tag = False
            else:
                pub_test_tag = False
                priv_test_tag = True

        while True:
            randir_narthan, randir_narthan_eng = generate_elvish_name()
            if randir_narthan not in nothrim:
                break
        #Generating Sindarin name and ensuring that it is unique

        nothrim.add(randir_narthan)
        #Adding name to set
        #obj_output_path = f'/Users/dylanmagill/python/lsst_sim_phot_outputs/{object_name}'
        if output_tag == 'test':
            obj_sub_run_output_path = sim_output_path + '/' + randir_narthan
            os.makedirs(obj_sub_run_output_path, exist_ok=True)
        #Making folder to store outputs

        #####   Generate SNCosmo Model   #####

        ###  Attempting to implement k-correction  ###

        ## Move LSST redshift calculation to before SNCosmo

        ztf_lim_mag = 20.6
        lsst_lim_mag = 24.7
        #Define ZTF and LSST limiting magnitudes (r-band)

        u_lim = ab_to_uJy(23.9); g_lim = ab_to_uJy(25.0); r_lim = ab_to_uJy(24.7)
        i_lim = ab_to_uJy(24.0); z_lim = ab_to_uJy(23.3); y_lim = ab_to_uJy(22.1)
        #Defining LSST limiting fluxes for each band

        ztf_lim_flux = ab_to_uJy(ztf_lim_mag)
        lsst_lim_flux = ab_to_uJy(lsst_lim_mag)
        #Converting limiting mags to fluxes

        dist_ZTF = Distance(unit=u.pc, z = redshift, cosmology = WMAP9)
        dist_scatter = np.random.uniform(0.9,1.1)
        dist_LSST = dist_ZTF * np.sqrt(ztf_lim_flux/lsst_lim_flux) * dist_scatter
        try:
            lsst_redshift = z_at_value(WMAP9.luminosity_distance, dist_LSST)
            warnings.simplefilter("ignore", category=AstropyWarning)
        except Exception as e:
            fail_reason = 'Redshift calculation failed'
            print(fail_reason)
            fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                        "SpecType": SpecType,
                        "Fail Reason": fail_reason}])
            fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
            continue
        print(f'LSST Redshift - {lsst_redshift}')
        #Calculate ZTF distance and use to get LSST distance and LSST redshift

        dilation_factor = (1 + lsst_redshift) / (1 + float(redshift))
        gp_timescale_dilated = gp_timescale * dilation_factor
        ## Apply time dilation to the main band model at the ZTF redshift
        main_band_rise_start_lsst = main_band_rise_start * dilation_factor

        if SpecType == 'SN Ia':
            model = sncosmo.Model(source='hsiao')
            print('Model chosen - hsiao')
            model_name = 'hsiao'
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ia'
            Ia_num += 10

            k_corr_model = sncosmo.Model(source='hsiao')
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model needed for k-correction

            #Generating model and setting parameters for SN Ia

        if SpecType == 'SN Ia-91T-like':
            model = sncosmo.Model(source='nugent-sn91t')
            #Model didn't survive k-corrections. Switching to Hsiao
            #model = sncosmo.Model(source='hsiao')
            print('Model chosen - nugent-sn91t')
            model_name = 'nugent-sn91t'
            sn91t_start = main_band_peak_time - 15
            sn91t_start_lsst = sn91t_start * dilation_factor
            model.set(z=redshift, t0=sn91t_start, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ia'
            Ia91T_num += 10

            k_corr_model = sncosmo.Model(source='nugent-sn91t')
            k_corr_model.set(z=lsst_redshift, t0=sn91t_start_lsst, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ia-91T

        if SpecType == 'SN Ia-91bg-like':
            model = sncosmo.Model(source='nugent-sn91bg')
            #Model didn't survive k-corrections. Switching to Hsiao
            #model = sncosmo.Model(source='hsiao')
            print('Model chosen - nugent-sn91bg')
            model_name = 'nugent-sn91bg'
            sn91bg_start = main_band_peak_time - 10
            sn91bg_start_lsst = sn91bg_start * dilation_factor
            model.set(z=redshift, t0=sn91bg_start, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ia'
            Ia91bg_num += 10

            k_corr_model = sncosmo.Model(source='nugent-sn91bg')
            k_corr_model.set(z=lsst_redshift, t0=sn91bg_start_lsst, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ia-91bg

        if SpecType == 'SN Iax[02cx-like]':
            model = sncosmo.Model(source='nugent-sn91t')
            #Model didn't survive k-corrections. Switching to Hsiao
            #model = sncosmo.Model(source='hsiao')            
            print('Model chosen - nugent-sn91t')
            model_name = 'nugent-sn91t'
            sn91t_start = main_band_peak_time - 15
            sn91t_start_lsst = sn91t_start * dilation_factor
            model.set(z=redshift, t0=sn91t_start, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ia'
            Ia02cx_num += 10

            k_corr_model = sncosmo.Model(source='nugent-sn91t')
            k_corr_model.set(z=lsst_redshift, t0=sn91t_start_lsst, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ia-02cx - model suggested by Shubham (only accurate within a month of explosion)

        if SpecType == 'SN Ia-02es-like':
            model = sncosmo.Model(source='nugent-sn91bg')
            #Model didn't survive k-corrections. Switching to Hsiao
            #model = sncosmo.Model(source='hsiao')
            print('Model chosen - nugent-sn91bg')
            model_name = 'nugent-sn91bg'
            sn91bg_start = main_band_peak_time - 10
            sn91bg_start_lsst = sn91bg_start * dilation_factor
            model.set(z=redshift, t0=sn91bg_start, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ia'
            Ia02es_num += 10

            k_corr_model = sncosmo.Model(source='nugent-sn91bg')
            k_corr_model.set(z=lsst_redshift, t0=sn91bg_start_lsst, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ia-02es - model suggested by Shubham

        if SpecType == 'SN Ia-pec':
            model = sncosmo.Model(source='hsiao')
            print('Model chosen - hsiao')
            model_name = 'hsiao'
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ia'
            Iapec_num += 10

            k_corr_model = sncosmo.Model(source='hsiao')
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
        #Generating model and setting parameters for SN Ia-pec [Shubham says nothing will be particularly great for this - so sticking with hsiao for now]

        if SpecType == 'SN Ib':
            Ib_model_list = ['v19-2004gq', 'v19-2007uy', 'v19-2012au']
            Ib_source = random.choice(Ib_model_list)
            #Ib_source = 'v19-2004gq'
            #Ib_source = 'v19-2007uy'
            #Ib_source = 'v19-2012au'
            print('Model chosen - ', Ib_source)
            model_name = Ib_source
            model = sncosmo.Model(source=Ib_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ib'
            Ib_num += 10

            k_corr_model = sncosmo.Model(source=Ib_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ib
            #Cut models : 'v19-2004gv', 'v19-2008d', 'v19-2009jf' 'v19-iptf13bvn'

    #if SpecType == 'SN Ibn':
    #    model = sncosmo.Model(source='hsiao')
    #    print('Model chosen - nugent-sn91bg')
    #    model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
    #    print(model.parameters)
        #Generating model and setting parameters for SN Ibn [placeholder - no clear model in SNCosmo]

        if SpecType == 'SN Ib/c':
            Ibc_model_list = ['v19-2005bf', 'v19-2006ep']
            Ibc_source = random.choice(Ibc_model_list)
            #Ibc_source = 'v19-2005bf'
            #Ibc_source = 'v19-2006ep'
            print('Model chosen - ', Ibc_source)
            model_name = Ibc_source
            model = sncosmo.Model(source=Ibc_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ib'
            Ibc_num += 10

            k_corr_model = sncosmo.Model(source=Ibc_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ib/c
            #Cut models : 'v19-2005hg', 'v19-2009iz'

        if SpecType == 'SN Ib-pec':
            model = sncosmo.Model(source='v19-2007uy')
            print('Model chosen - v19-2007uy')
            model_name = 'v192007uy'
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ib'
            Ibpec_num += 10

            k_corr_model = sncosmo.Model(source='v19-2007uy')
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ib-pec

        if SpecType == 'SN Ic':
            Ic_model_list = ['v19-2004gt', 'v19-2011bm']
            Ic_source = random.choice(Ic_model_list)
            #Ic_source = 'v19-2004gt'
            #Ic_source = 'v19-2011bm'
            print('Model chosen - ', Ic_source)
            model_name = Ic_source
            model = sncosmo.Model(source=Ic_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ic'
            Ic_num += 10

            k_corr_model = sncosmo.Model(source=Ic_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ic
            #Cut models : 'v19-1994i', 'v19-2007gr', 'v19-2013ge' - tentatively keeping 2011bm

        if SpecType == 'SN Ic-BL':
            IcBL_model_list = ['v19-1998bw', 'v19-2002ap', 'v19-2007ru']
            IcBL_source = random.choice(IcBL_model_list)
            #IcBL_source = 'v19-1998bw'
            #IcBL_source = 'v19-2002ap'
            #IcBL_source = 'v19-2007ru'
            print('Model chosen - ', IcBL_source)
            model_name = IcBL_source
            model = sncosmo.Model(source=IcBL_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'Ic'
            IcBL_num += 10

            k_corr_model = sncosmo.Model(source=IcBL_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN Ic-BL
            #Cut models : 'v19-2006aj', 'v19-2009bb', 'v19-2012ap'

        if SpecType == 'SN II' or SpecType == 'SN IIP':
            II_model_list = ['v19-2009kr', 'v19-2013ab', 'v19-2013am', 
                        'v19-1987a', 'v19-2009ib', 'v19-2016x']
            II_source = random.choice(II_model_list)
            #II_source = 'v19-2009kr'
            #II_source = 'v19-2013ab'
            #II_source = 'v19-2013am'
            #II_source = 'v19-1987a'
            #II_source = 'v19-2009ib'
            #II_source = 'v19-2016x'
            print('Model chosen - ', II_source)
            model_name = II_source
            model = sncosmo.Model(source=II_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'II'
            II_num += 10

            k_corr_model = sncosmo.Model(source=II_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN II
            #Cut models : 'v19-asassn14jb', 'v19-asassn15oz', 'v19-2015bkv', 'v19-2014g', 'v19-2013ej', 'v19-2013by',
            # 'v19-2012aw', 'v19-2012a', 'v19-2009n', 'v19-2009dd', 'v19-2009bw', 'v19-2008in', 'v19-2008bj', 'v19-2007od',
            # 'v19-2004et', 'v19-1999em' - tentatively keeping 1987a, 2013ab, 2013am and 2016x

        if SpecType == 'SN IIb':
            IIb_model_list = ['v19-1999dn', 'v19-2006t','v19-2008bo',
                        'v19-2011dh', 'v19-2011ei', 'v19-2011fu', 'v19-2011hs',]
            IIb_source = random.choice(IIb_model_list)
            #IIb_source = 'v19-1999dn'
            #IIb_source = 'v19-2006t'
            #IIb_source = 'v19-2008aq'
            #IIb_source = 'v19-2008bo'
            #IIb_source = 'v19-2011dh'
            #IIb_source = 'v19-2011ei'
            #IIb_source = 'v19-2011fu'
            #IIb_source = 'v19-2011hs'
            print('Model chosen - ', IIb_source)
            model_name = IIb_source
            model = sncosmo.Model(source=IIb_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'II'
            IIb_num += 10

            k_corr_model = sncosmo.Model(source=IIb_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN IIb
            #Cut models : 'v19-1993j', 'v19-2008ax', 'v19-2013df', 'v19-2016gkg', 'v19-2008aq',  - tentatively keeping 2011ei, 2011fu and 2011hs

        if SpecType == 'SN IIn':
            IIn_model_list = ['v19-2006aa', 'v19-2011ht']
            IIn_source = random.choice(IIn_model_list)
            #IIn_source = 'v19-2006aa'
            #IIn_source = 'v19-2011ht'
            print('Model chosen - ', IIn_source)
            model_name = IIn_source
            model = sncosmo.Model(source=IIn_source)
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'II'
            IIn_num += 10

            k_corr_model = sncosmo.Model(source=IIn_source)
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SN IIn
            #Cut models : 'v19-2007pk', 'v19-2008fq', 'v19-2010al', 'v19-2009ip'

        if SpecType == 'SLSN-I':
            #SLSNI_model_list = ['v19-2006aj', 'v19-2007ru','v19-2011bm']
            #SLSNI_source = random.choice(SLSNI_model_list)
            #SLSNI_source = 'v19-2006aj'
            #SLSNI_source = 'v19-2007ru'
            #SLSNI_source = 'v19-2011bm'
            #print('Model chosen - ', SLSNI_source)            
            #model_name = SLSNI_source
            #model = sncosmo.Model(source=SLSNI_source)
            #model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)

            SLSN_start_temp = np.random.uniform(10000,15000)
            SLSN_end_temp = np.random.uniform(5000,8000)
            SLSN_delta_T = (SLSN_start_temp - SLSN_end_temp) / np.random.uniform(30,100)
            
            model = sncosmo.Model(source=dT_dt_Blackbody_SLSN())
            model_name = 'dT_dt_Blackbody_SLSN'
            model.set(amplitude=2e-16, temperature=SLSN_start_temp, dT_dt = SLSN_delta_T, z=redshift)                        
            print('Model chosen - ', model_name)            

            #print(model.parameters)
            spec_tag = 'SLSN'
            SLSNI_num += 10

            k_corr_model = sncosmo.Model(source=dT_dt_Blackbody_SLSN())
            k_corr_model.set(amplitude=2e-16, temperature=SLSN_start_temp, dT_dt = SLSN_delta_T, z=lsst_redshift)
            #Generating model and setting parameters for SLSN-I [2006aj and 2007ru - bluest Ic-BLs]
            #Switched to a blackbody model based on Nicholl et al 2017 and Gomez et al 2024

        if SpecType == 'SLSN-II':
            model = sncosmo.Model(source='v19-2011ht')
            print('Model chosen - v19-2011ht')
            model_name = 'v19-2011ht'
            model.set(z=redshift, t0=main_band_peak_time, amplitude=main_band_peak)
            #print(model.parameters)
            spec_tag = 'SLSN'
            SLSNII_num += 10

            k_corr_model = sncosmo.Model(source='v19-2011ht')
            k_corr_model.set(z=lsst_redshift, t0=main_band_peak_time, amplitude = main_band_peak)
            #Generating model and setting parameters for SLSN-II [brightest smooth IIn model]

        if SpecType == 'TDE':
            tde_temp_path = '/Users/dylanmagill/python/tde_temps.csv'
            tde_cols = ['TDE_name', 'log_bb_temp', 'temp']
            tde_temp_df = pd.read_csv(tde_temp_path, sep=",", skiprows=0, usecols=tde_cols)
            tde_temp_list = tde_temp_df.iloc[:,2]

            tde_model_temp = random.choice(tde_temp_list)
            #tde_model_temp = 30000
        
            print('TDE temperature - ', tde_model_temp)
            #Selecting random temperature for TDE from Vysakh's list

            delta_T_json_path = '/Users/dylanmagill/python/TDE_temperatures.json'
            delta_T_df = pd.read_json(delta_T_json_path, orient='index')
            min_temp_diff_index = np.abs(delta_T_df['T_peak'] - tde_model_temp).argmin()
            delta_T_info = delta_T_df.iloc[min_temp_diff_index]
            delta_T = delta_T_info['T_dT']
            print(f'dT/dt value = {delta_T}')

            #counts, bins = np.histogram(tde_temp_list,bins=15)
            #fig, axes = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
            #axes.stairs(counts,bins)
            #axes.set_xlabel('Temperature - K')
            #axes.set_ylabel('Frequency')
            #plt.savefig('EAS_tde_temps_hist.pdf')
            #plt.show()
            #Plotting histogram of TDE temperatures

            #model = sncosmo.Model(source=Blackbody())
            #model_name = 'Blackbody'
            #model.set(amplitude=2e-16, temperature=tde_model_temp, z=redshift)
            #print(model.parameters)

            #k_corr_model = sncosmo.Model(source=Blackbody())
            #k_corr_model.set(amplitude=2e-16, temperature=tde_model_temp, z=lsst_redshift)
            # ^ Original constant temperature model

            model = sncosmo.Model(source=dT_dt_Blackbody())
            model_name = 'dT_dt_Blackbody'
            model.set(amplitude=2e-16, temperature=tde_model_temp, dT_dt = delta_T, z=redshift)

            print('ZTF model created')

            spec_tag = 'TDE'
            TDE_num += 10

            k_corr_model = sncosmo.Model(source=dT_dt_Blackbody())
            k_corr_model.set(amplitude=2e-16, temperature=tde_model_temp, dT_dt = delta_T, z=lsst_redshift)
            print('LSST model created')
            # ^ New temperature change model

            #Generating model and setting parameters for TDEs

        if SpecType == 'AGN':
            agn_temp_path = '/Users/dylanmagill/python/agn_temps.csv'
            agn_cols = ['AGN_name', 'log_bb_temp', 'temp']
            agn_temp_df = pd.read_csv(agn_temp_path, sep=",", skiprows=0, usecols=agn_cols)
            agn_temp_list = agn_temp_df.iloc[:,2]

            #counts, bins = np.histogram(agn_temp_list,bins=15)
            #fig, axes = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
            #axes.stairs(counts,bins)
            #axes.set_xlabel('Temperature - K')
            #axes.set_ylabel('Frequency')
            #plt.savefig('EAS_agn_temps_hist.pdf')
            #plt.show()
            #Plotting histogram of AGN temperatures

            agn_model_temp = random.choice(agn_temp_list)
            print('AGN temperature - ', agn_model_temp)
            #Selecting random temperature for AGN from Vysakh's list
            model = sncosmo.Model(source=Blackbody())
            model_name = 'Blackbody'
            model.set(amplitude=2e-16, temperature=agn_model_temp, z=redshift)
            #print(model.parameters)
            spec_tag = 'AGN'
            AGN_num += 10

            k_corr_model = sncosmo.Model(source=Blackbody())
            k_corr_model.set(amplitude=2e-16, temperature=agn_model_temp, z=lsst_redshift)
            #Generating model and setting parameters for AGN

    #Some subtypes I'm not sure of = Varstar, Galaxy, CV, QSO, Ia-CSM
    #Some are defined as 'Other'? - Probably need to check what they are manually on TNS
    #Also an SN I? (ZTF22abieqzx, ZTF22abmuvgq)

        if output_tag == 'test':
            if main_band_tag == 'r':
                plt.errorbar(main_band_time, main_band_flux, yerr = main_band_error, label = 'observed_r', color = 'red', fmt= 'o')
                plt.plot(gp_timescale,main_band_flux_gp, label = 'Gaussian_r', color = 'orange')
                plt.xlabel('Days - Relative to Peak')
                plt.ylabel('Flux - μJy')
                plt.legend()
            #    plt.xlim(-750,750)
                plt.savefig(obj_output_path+'/obs_vs_GP.pdf')
                plt.close()
            #    plt.show()
                #Plotting against main band data to test model

            if main_band_tag == 'g':
                plt.plot(main_band_time,main_band_flux,label='observed_g', color = filter_colours['g'])
                plt.plot(gp_timescale,main_band_flux_gp, label = 'g-band GP Predictions', color = 'orange')
                plt.legend()
                plt.savefig(obj_output_path+'/obs_vs_GP.pdf')
                plt.close()
             #   plt.show()
                #Plotting against main band data to test model

    #####   Determine Colour Differences   #####

        try:
            print('Starting band model creation')
            #model_u = model.bandmag('lsstu','ab',time=gp_timescale)
            model_g = model.bandmag('lsstg','ab',time=gp_timescale)
            print('ZTF g model created')
            model_r = model.bandmag('lsstr','ab',time=gp_timescale)
            print('ZTF r model created')

            #plt.plot(gp_timescale,model_g,label = 'g' ,color = filter_colours['g'])
            #plt.plot(gp_timescale,model_r,label = 'r', color = filter_colours['r'])
            #plt.gca().invert_yaxis()
            #plt.legend()
            #plt.savefig('snc_models_ztf.pdf')
            #plt.show()
            #model_i = model.bandmag('lssti','ab',time=gp_timescale)
            #model_z = model.bandmag('lsstz','ab',time=gp_timescale)
            #model_y = model.bandmag('lssty','ab',time=gp_timescale)
            ## Calculate SNCosmo models for g and r band using ZTF redshift as normal - done above ^^

            k_corr_model_u = k_corr_model.bandmag('lsstu','ab',time=gp_timescale_dilated)
            #if SpecType == 'SN Ia-91T-like' or SpecType == 'SN Ia-91bg-like':
            #    from scipy.ndimage import gaussian_filter1d
            #    k_corr_model_u = gaussian_filter1d(k_corr_model_u, sigma=2)
            k_corr_model_g = k_corr_model.bandmag('lsstg','ab',time=gp_timescale_dilated)
            k_corr_model_r = k_corr_model.bandmag('lsstr','ab',time=gp_timescale_dilated)
            k_corr_model_i = k_corr_model.bandmag('lssti','ab',time=gp_timescale_dilated)
            k_corr_model_z = k_corr_model.bandmag('lsstz','ab',time=gp_timescale_dilated)
            k_corr_model_y = k_corr_model.bandmag('lssty','ab',time=gp_timescale_dilated)

            # plt.plot(gp_timescale,k_corr_model_u,label = 'u',color = filter_colours['u'])
            # plt.plot(gp_timescale,k_corr_model_g,label = 'g',color = filter_colours['g'])
            # plt.plot(gp_timescale,k_corr_model_i,label = 'i',color = filter_colours['i'])
            # plt.plot(gp_timescale,k_corr_model_r,label = 'r',color = filter_colours['r'])
            # plt.plot(gp_timescale,k_corr_model_z,label = 'z',color = filter_colours['z'])
            # plt.plot(gp_timescale,k_corr_model_y,label = 'y',color = filter_colours['y'])
            # plt.gca().invert_yaxis()
            # plt.legend()
            # plt.savefig('snc_models_lsst.pdf')
            # plt.show()
            #plt.close()
            ## Calculate SNCosmo models for all LSST bands at the LSST redshift

        except Exception as e:
            fail_reason = "SNCosmo fit failed"
            print(fail_reason)
            fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                                    "SpecType": SpecType,
                                    "Fail Reason": fail_reason}])
            fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
            continue 
        #Wrapped in try statement to prevent crashing
        #Generating models for other bands

        lower_colour_max = (-50) * (1 + lsst_redshift)
        upper_colour_max = 150 * (1 + lsst_redshift)
        if SpecType == 'SN Ic-BL' or spec_tag == 'SLSN' or SpecType == 'SN IIn':
            lower_colour_max = (-100) * (1 + lsst_redshift)
            upper_colour_max = 300 * (1 + lsst_redshift)
        if spec_tag == 'TDE':
            lower_colour_max = (-100) * (1 + lsst_redshift)
            upper_colour_max = 400 * (1 + lsst_redshift)
        print(f'Colour Boundaries - {lower_colour_max}, {upper_colour_max}')
        colour_range_mask = (gp_timescale_dilated < lower_colour_max) | (gp_timescale_dilated > upper_colour_max)

        # def colour_clean(colour, mask):
        #     colour = colour.copy()
        #     valid_idx = np.where(~mask)[0]   # indices where mask is False (valid)
        #     if len(valid_idx) == 0:
        #         return colour  # nothing valid, just return as is

        #     first_valid = valid_idx[0]
        #     last_valid = valid_idx[-1]

        #     # Before first valid - set to first valid value
        #     colour[:first_valid] = colour[first_valid]
        #     # After last valid - set to last valid value
        #     colour[last_valid+1:] = colour[last_valid]

        #     return colour


        if main_band_tag == 'r':
            #model_r = np.interp(gp_timescale, gp_timescale_dilated, model_r)
            #Rescaling time axis of ZTF model to line up with LSST models

            r_minus_u = model_r - k_corr_model_u
            r_minus_g = model_r - k_corr_model_g
            r_minus_r = model_r - k_corr_model_r
            r_minus_i = model_r - k_corr_model_i
            r_minus_z = model_r - k_corr_model_z
            r_minus_y = model_r - k_corr_model_y

            # if SpecType == 'SN Ia-91T-like' or SpecType == 'SN Ia-91bg-like':
            #     r_minus_u = r_minus_u / 2
            #     r_minus_g = r_minus_g / 2
            #     r_minus_r = r_minus_r / 2
            #     r_minus_i = r_minus_i / 2
            #     r_minus_z = r_minus_z / 2
            #     r_minus_y = r_minus_y / 2

            # r_minus_u_mask = abs(r_minus_u) > 5
            # r_minus_u[r_minus_u_mask] = 0
            # r_minus_g_mask = abs(r_minus_g) > 5
            # r_minus_g[r_minus_g_mask] = 0
            # r_minus_r_mask = abs(r_minus_r) > 5
            # r_minus_r[r_minus_r_mask] = 0
            # r_minus_i_mask = abs(r_minus_i) > 5
            # r_minus_i[r_minus_i_mask] = 0
            # r_minus_z_mask = abs(r_minus_z) > 5
            # r_minus_z[r_minus_z_mask] = 0
            # r_minus_y_mask = abs(r_minus_y) > 5
            # r_minus_y[r_minus_y_mask] = 0        
            print(spec_tag)
            if spec_tag != 'AGN':
                #colour_time_mask_min = gp_timescale_dilated < (-50)
                #colour_time_mask_max = gp_timescale_dilated > 100

                # r_minus_u[colour_time_mask_min] = 0
                # r_minus_u[colour_time_mask_max] = 0
                r_minus_u[colour_range_mask] = 0
                #r_minus_u = colour_clean(r_minus_u, colour_range_mask)

                # r_minus_g[colour_time_mask_min] = 0
                # r_minus_g[colour_time_mask_max] = 0
                r_minus_g[colour_range_mask] = 0
                #r_minus_g = colour_clean(r_minus_g, colour_range_mask)

                # r_minus_r[colour_time_mask_min] = 0
                # r_minus_r[colour_time_mask_max] = 0
                r_minus_r[colour_range_mask] = 0
                #r_minus_r = colour_clean(r_minus_r, colour_range_mask)

                # r_minus_i[colour_time_mask_min] = 0
                # r_minus_i[colour_time_mask_max] = 0
                r_minus_i[colour_range_mask] = 0
                #r_minus_i = colour_clean(r_minus_i, colour_range_mask)

                # r_minus_z[colour_time_mask_min] = 0
                # r_minus_z[colour_time_mask_max] = 0
                r_minus_z[colour_range_mask] = 0
                #r_minus_z = colour_clean(r_minus_z, colour_range_mask)

                # r_minus_y[colour_time_mask_min] = 0
                # r_minus_y[colour_time_mask_max] = 0    
                r_minus_y[colour_range_mask] = 0
                #r_minus_y = colour_clean(r_minus_y, colour_range_mask)


            # plt.plot(gp_timescale_dilated,r_minus_u,label='r-u',color=filter_colours['u'])
            # plt.plot(gp_timescale_dilated,r_minus_g,label='r-g',color=filter_colours['g'])
            # plt.plot(gp_timescale_dilated,r_minus_r,label='r-r',color=filter_colours['r'])
            # plt.plot(gp_timescale_dilated,r_minus_i,label='r-i',color=filter_colours['i'])
            # plt.plot(gp_timescale_dilated,r_minus_z,label='r-z',color=filter_colours['z'])
            # plt.plot(gp_timescale_dilated,r_minus_y,label='r-y',color=filter_colours['y'])
            # plt.legend()
            # plt.show()
        #Calculating colour differences if r band

        if main_band_tag == 'g':
            #model_g = np.interp(gp_timescale, gp_timescale_dilated, model_g)
            #Rescaling time axis of ZTF model to line up with LSST models

            g_minus_u = model_g - k_corr_model_u
            g_minus_g = model_g - k_corr_model_g
            g_minus_r = model_g - k_corr_model_r
            g_minus_i = model_g - k_corr_model_i
            g_minus_z = model_g - k_corr_model_z
            g_minus_y = model_g - k_corr_model_y

            # g_minus_u_mask = abs(g_minus_u) > 5
            # g_minus_u[g_minus_u_mask] = 0
            # g_minus_g_mask = abs(g_minus_g) > 5
            # g_minus_g[g_minus_g_mask] = 0
            # g_minus_r_mask = abs(g_minus_r) > 5
            # g_minus_r[g_minus_r_mask] = 0
            # g_minus_i_mask = abs(g_minus_i) > 5
            # g_minus_i[g_minus_i_mask] = 0
            # g_minus_z_mask = abs(g_minus_z) > 5
            # g_minus_z[g_minus_z_mask] = 0
            # g_minus_y_mask = abs(g_minus_y) > 5
            # g_minus_y[g_minus_y_mask] = 0        

            print(spec_tag)
            if spec_tag != 'AGN':
                #colour_time_mask_min = gp_timescale_dilated < (-50)
                #colour_time_mask_max = gp_timescale_dilated > 100

                #print(g_minus_u[colour_time_mask_min])
                #g_minus_u[colour_time_mask_min] = 0
                #g_minus_u[colour_time_mask_max] = 0
                g_minus_u[colour_range_mask] = 0
                #g_minus_u = colour_clean(g_minus_u, colour_range_mask)

                #g_minus_g[colour_time_mask_min] = 0
                #g_minus_g[colour_time_mask_max] = 0
                g_minus_g[colour_range_mask] = 0
                #g_minus_g = colour_clean(g_minus_g, colour_range_mask)

                #g_minus_r[colour_time_mask_min] = 0
                #g_minus_r[colour_time_mask_max] = 0
                g_minus_r[colour_range_mask] = 0
                #g_minus_r = colour_clean(g_minus_r, colour_range_mask)

                #g_minus_i[colour_time_mask_min] = 0
                #g_minus_i[colour_time_mask_max] = 0
                g_minus_i[colour_range_mask] = 0
                #g_minus_i = colour_clean(g_minus_i, colour_range_mask)

                #g_minus_z[colour_time_mask_min] = 0
                #g_minus_z[colour_time_mask_max] = 0
                g_minus_z[colour_range_mask] = 0
                #g_minus_z = colour_clean(g_minus_z, colour_range_mask)

                #g_minus_y[colour_time_mask_min] = 0
                #g_minus_y[colour_time_mask_max] = 0    
                g_minus_y[colour_range_mask] = 0
                #g_minus_y = colour_clean(g_minus_y, colour_range_mask)

            # plt.plot(gp_timescale_dilated,g_minus_u,label='r-u',color=filter_colours['u'])
            # plt.plot(gp_timescale_dilated,g_minus_g,label='r-g',color=filter_colours['g'])
            # plt.plot(gp_timescale_dilated,g_minus_r,label='r-r',color=filter_colours['r'])
            # plt.plot(gp_timescale_dilated,g_minus_i,label='r-i',color=filter_colours['i'])
            # plt.plot(gp_timescale_dilated,g_minus_z,label='r-z',color=filter_colours['z'])
            # plt.plot(gp_timescale_dilated,g_minus_y,label='r-y',color=filter_colours['y'])
            # plt.legend()
            # plt.show()
        #Calculating colour differences if g band

        ## Calculate colour + k-correction curves : r_ztf - u_lsst, r_ztf - g_lsst, r_ztf - r_lsst etc. - done above ^^

        #####   Construct Points In Other Bands   #####

        def colour_convert (input_flux, colour_diff):
            conv_flux = input_flux * 10**(colour_diff/2.5)
            return conv_flux
        #Defining a function to carry out the colour conversions

        #print(main_band_flux_gp)

        if main_band_tag == 'r':
            sim_u_flux = colour_convert(main_band_flux_gp,r_minus_u)
            sim_g_flux = colour_convert(main_band_flux_gp,r_minus_g)
            main_band_flux_gp_corr = colour_convert(main_band_flux_gp,r_minus_r)
            sim_i_flux = colour_convert(main_band_flux_gp,r_minus_i)
            sim_z_flux = colour_convert(main_band_flux_gp,r_minus_z)
            sim_y_flux = colour_convert(main_band_flux_gp,r_minus_y)
        if main_band_tag == 'g':
            sim_u_flux = colour_convert(main_band_flux_gp,g_minus_u)
            main_band_flux_gp_corr = colour_convert(main_band_flux_gp,g_minus_g)
            sim_r_flux = colour_convert(main_band_flux_gp,g_minus_r)
            sim_i_flux = colour_convert(main_band_flux_gp,g_minus_i)
            sim_z_flux = colour_convert(main_band_flux_gp,g_minus_z)
            sim_y_flux = colour_convert(main_band_flux_gp,g_minus_y)
        #Using function to generate points in other bands

        ## Apply colour + k-correction curves to the data to get LSST models in all bands - done above ^^

        if output_tag == 'test':
            if main_band_tag == 'r':
                plt.plot(gp_timescale_dilated,sim_u_flux,label='u',color = filter_colours['u'])
                plt.plot(gp_timescale_dilated,sim_g_flux,label='g', color = filter_colours['g'])
                plt.plot(gp_timescale_dilated,main_band_flux_gp_corr,label='r', color = filter_colours['r'])
                plt.plot(gp_timescale_dilated,sim_i_flux,label='i', color = filter_colours['i'])
                plt.plot(gp_timescale_dilated,sim_z_flux,label='z', color = filter_colours['z'])
                plt.plot(gp_timescale_dilated,sim_y_flux,label='y', color = filter_colours['y'])
                plt.xlabel('Days - Relative to Peak')
                plt.ylabel('Flux - μJy')
                #plt.xlim(-750,750)
                plt.legend()
                plt.savefig(obj_output_path+'/6band_contiguous.pdf')
                plt.close()
                #plt.show()
            if main_band_tag == 'g':
                plt.plot(gp_timescale_dilated,sim_u_flux,label='u', color = filter_colours['u'])
                plt.plot(gp_timescale_dilated,main_band_flux_gp_corr,label='g', color = filter_colours['g'])
                plt.plot(gp_timescale_dilated,sim_r_flux,label='r', color = filter_colours['r'])
                plt.plot(gp_timescale_dilated,sim_i_flux,label='i', color = filter_colours['i'])
                plt.plot(gp_timescale_dilated,sim_z_flux,label='z', color = filter_colours['z'])
                plt.plot(gp_timescale_dilated,sim_y_flux,label='y', color = filter_colours['y'])
                plt.xlabel('Days - Relative to Peak')
                plt.ylabel('Flux - μJy')
                plt.legend()
                plt.savefig(obj_output_path+'/6band_contiguous.pdf')
                plt.close()
                #plt.show()
            #Plotting to check lightcurve

        ##### LSST Redshift Determination & Flux Reddening  #####

        #if main_band_tag == 'r':
        #    pre_scale_sim_u_flux = sim_u_flux; pre_scale_sim_g_flux = sim_g_flux; pre_scale_main_band_flux = main_band_flux_gp_corr
        #    pre_scale_sim_i_flux = sim_i_flux; pre_scale_sim_z_flux = sim_z_flux; pre_scale_sim_y_flux = sim_y_flux
        #if main_band_tag == 'g':
        #    pre_scale_sim_u_flux = sim_u_flux; pre_scale_main_band_flux = main_band_flux_gp_corr; pre_scale_sim_r_flux = sim_r_flux
        #    pre_scale_sim_i_flux = sim_i_flux; pre_scale_sim_z_flux = sim_z_flux; pre_scale_sim_y_flux = sim_y_flux
        #Saving flux values before scaling to allow for creation of multiple lightcurves

        #if main_band_tag == 'r':
        #    sim_u_flux = pre_scale_sim_u_flux; sim_g_flux = pre_scale_sim_g_flux; main_band_flux_gp = pre_scale_main_band_flux
        #    sim_i_flux = pre_scale_sim_i_flux; sim_z_flux = pre_scale_sim_i_flux; sim_y_flux = pre_scale_sim_y_flux
        #if main_band_tag == 'g':
        #    sim_u_flux = pre_scale_sim_u_flux; main_band_flux_gp = pre_scale_main_band_flux; sim_r_flux = pre_scale_sim_r_flux
        #    sim_i_flux = pre_scale_sim_i_flux; sim_z_flux = pre_scale_sim_i_flux; sim_y_flux = pre_scale_sim_y_flux
        #Resetting fluxes to pre-scaling values

        def flux_reddening (flux,fudge):
            flux_lsst = flux * (lsst_lim_flux/ztf_lim_flux) * fudge
            return flux_lsst 
        #Defining function to scale LSST flux to same level above LSST limit as it was above ZTF limit
        #Fudge value allows for some random scatter in the peak magnitudes

        fudge = 1
        #Defining placeholder fudge value

        def fudge_function (spec_tag):
            if spec_tag == 'Ia':
                #fudge = 10 ** (-np.random.uniform(-0.784, 0.753) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.375)/2.5) 
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3)/2.5)
                print(f'Ia Fudge Value = {fudge}')
                #From Jack (0.75/2)

            if spec_tag == 'Ib':
                #fudge = 10 ** (-np.random.uniform(-0.7, 0.7) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.35) / 2.5)
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3) / 2.5)
                print(f'Ib Fudge Value = {fudge}')
                #From Thomas' Ibc paper (0.7/2)

            if spec_tag == 'Ic':
                #fudge = 10 ** (-np.random.uniform(-1.3, 0.6) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = -0.175, scale = 0.475) / 2.5)
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3) / 2.5)
                print(f'Ic Fudge Value = {fudge}')
                #From Thomas' Ibc paper (-1.3,0.6 / 2)

            if spec_tag == 'II':
                #fudge = 10 ** (-np.random.uniform(-1.01, 1.01) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.505) / 2.5)
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3) / 2.5)
                print(f'II Fudge Value = {fudge}')
                #From Anderson 2014 paper (1.01/2)

            if spec_tag == 'SLSN':
                #fudge = 10 ** (-np.random.uniform(-0.3, 0.6) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = 0.075, scale = 0.225) / 2.5)
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3) / 2.5)
                print(f'SLSN Fudge Value = {fudge}')
                #From Gomez 2024 paper

            if spec_tag == 'TDE':
                #fudge = 10 ** np.random.normal(0, 0.664)
                #fudge = 10 ** (-np.random.uniform(-2, 2) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.5) / 2.5)
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3) / 2.5)
                print(f'TDE Fudge Value = {fudge}')
                #From Yuhan's paper - from std of 0.664 over average of 44.1

            if spec_tag == 'AGN':
                #fudge = 10 ** (-np.random.uniform(-1, 1) / 2.5)
                #fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.5) / 2.5)
                fudge = 10 ** (-np.random.normal(loc = 0, scale = 0.3) / 2.5)                
                print(f'AGN Fudge Value = {fudge}')
                #From Lyke 2020 SDSS paper - plus/minus 1 mag
            
            return fudge
        #Defining function to create fudge value for a given type of object

        mag_safety_check = 0
        mag_safety_check_counter = 0
        mag_safety_check_tag = True
        #Setting placeholder value for safety check to constrain faint magnitudes

        if main_band_tag == 'r':
            while mag_safety_check < 2.5*r_lim:
                fudge = random.uniform(0.8,1.2)
                print(fudge)
                sim_u_flux = flux_reddening(sim_u_flux,fudge)
                sim_g_flux = flux_reddening(sim_g_flux,fudge)
                main_band_flux_gp_scaled = flux_reddening(main_band_flux_gp_corr,fudge)
                sim_i_flux = flux_reddening(sim_i_flux,fudge)
                sim_z_flux = flux_reddening(sim_z_flux,fudge)
                sim_y_flux = flux_reddening(sim_y_flux,fudge)
                mag_safety_check = np.max(main_band_flux_gp_scaled)
                mag_safety_check_counter += 1
                if mag_safety_check_counter > 5 and spec_tag != 'TDE':
                    mag_safety_check_tag = False
                    break
                if mag_safety_check_counter > 25 and spec_tag == 'TDE':
                    mag_safety_check_tag = False
                    break
        if main_band_tag == 'g':
            while mag_safety_check < 2.5*g_lim:
                fudge = random.uniform(0.8,1.2)
                sim_u_flux = flux_reddening(sim_u_flux,fudge)
                main_band_flux_gp_scaled = flux_reddening(main_band_flux_gp_corr,fudge)
                sim_r_flux = flux_reddening(sim_r_flux,fudge)
                sim_z_flux = flux_reddening(sim_z_flux,fudge)
                sim_i_flux = flux_reddening(sim_i_flux,fudge)
                sim_y_flux = flux_reddening(sim_y_flux,fudge)
                mag_safety_check = np.max(main_band_flux_gp_scaled)
                mag_safety_check_counter += 1
                if mag_safety_check_counter > 5 and spec_tag != 'TDE':
                    mag_safety_check_tag = False
                    break
                if mag_safety_check_counter > 25 and spec_tag == 'TDE':
                    mag_safety_check_tag = False
                    break
        #Scaling flux to LSST levels. With while loop safety check to constrain faint magnitudes

        if mag_safety_check_tag == False:
            print(f"Mag safety check repeatedly failed — skipping object.")
            fail_reason = 'Mag safety check failed'
            fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                        "SpecType": SpecType,
                        "Fail Reason": fail_reason}])
            fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
            continue

        print('ZTF Distance - ,', dist_ZTF)
        print('LSST Distance - ', dist_LSST)
        print('Distance Diff - ', dist_LSST - dist_ZTF)
        print('LSST Redshift - ', lsst_redshift)
        #Print statements to check values

        #####   Implementing LSST Cadence   #####

        lsst_e_date = 62000; lsst_l_date = 64000
        #Defining acceptable range of LSST dates

        lsst_lower_dec_limit = -75; lsst_upper_dec_limit = 15; lsst_extraupper_dec_limit = 35
        #List acceptable RA and Dec values - unclear what upper limit actually is, will check with Matt

        if obj_Dec > lsst_lower_dec_limit and obj_Dec < lsst_upper_dec_limit:
            sim_obj_ra = obj_RA; sim_obj_dec = obj_Dec
        else:
            sim_obj_ra = random.randint(0,360)
            sim_obj_dec = random.randint(lsst_lower_dec_limit, lsst_upper_dec_limit)
        #Check if exisitng RA and Dec is within range - if not randomly generate new RA and Dec within range

        #DDF_dict_list = []
        #ddf_dict_1 = {'Name' : 'ELAISS1',
        #        'RA' : 9.45,
        #        'Dec' : -44.02}
        #DDF_dict_list.append(ddf_dict_1)
        #ddf_dict_2 = {'Name' : 'XMM_LSS',
        #        'RA' : 35.57,
        #        'Dec' : -4.82}
        #DDF_dict_list.append(ddf_dict_2)
        #ddf_dict_3 = {'Name' : 'ECDFS',
        #        'RA' : 52.98,
        #        'Dec' : -28.12}
        #DDF_dict_list.append(ddf_dict_3)
        #ddf_dict_4 = {'Name' : 'COSMOS',
        #        'RA' : 150.11,
        #        'Dec' : 2.23}
        #DDF_dict_list.append(ddf_dict_4)
        #ddf_dict_5 = {'Name' : 'EDFS_a',
        #        'RA' : 58.9,
        #        'Dec' : -49.32}
        #DDF_dict_list.append(ddf_dict_5)
        #ddf_dict_6 = {'Name' : 'EDFS_b',
        #        'RA' : 63.6,
        #        'Dec' : -47.6}
        #DDF_dict_list.append(ddf_dict_6)
        #Setting values for Deep Drilling Fields (DDFs)

        #if DDF_tag == True :
        #    DDF_target = random.choice(DDF_dict_list)
        #    print("DDF Selected",DDF_target.get('Name'))
        #    sim_obj_ra = DDF_target.get('RA')
        #    sim_obj_dec = DDF_target.get('Dec')
        #Assigning a DDF RA and Dec if choosing to run DDF lightcurves

        # def get_match(band_visits, gp_time, gp_flux, err_list, tol=0.5):
        #     matched_times = []
        #     matched_flux = []
        #     matched_err = []
        #     for visit in band_visits:
        #         matches = np.where(np.abs(gp_time - visit) <= tol)[0]
        #         if matches.size > 0:
        #             idx = matches[0]
        #             matched_times.append(gp_time[idx])
        #             matched_flux.append(gp_flux[idx])
        #             matched_err.append(err_list[band_visits.index(visit)])
        #     return np.array(matched_times), np.array(matched_flux), np.array(matched_err)
        # #Defining function to find LSST cadence for chosen object

        # def get_match(band_visits, gp_time, gp_flux, err_list, tol=0.5):
        #     matched_times, matched_flux, matched_err = [], [], []
        #     band_visits = np.unique(np.round(band_visits, 4))
        #     for i, visit in enumerate(band_visits):
        #         idx = np.argmin(np.abs(gp_time - visit))
        #         if np.abs(gp_time[idx] - visit) <= tol:
        #             matched_times.append(gp_time[idx])
        #             matched_flux.append(gp_flux[idx])
        #             matched_err.append(err_list[i])
        #     return np.array(matched_times), np.array(matched_flux), np.array(matched_err)

        def get_match(band_visits, gp_time, gp_flux, err_list, tol=0.5):
            matched = {}
            
            # Ensure consistent rounding
            band_visits = np.round(band_visits, 4)
            gp_time = np.round(gp_time, 4)
            
            # Deduplicate visits while preserving errors
            band_visits, unique_idx = np.unique(band_visits, return_index=True)
            err_list = np.array(err_list)[unique_idx]

            for i, visit in enumerate(band_visits):
                # Find the GP point closest in time
                idx = np.argmin(np.abs(gp_time - visit))
                if np.abs(gp_time[idx] - visit) <= tol:
                    key = float(np.round(gp_time[idx], 4))
                    if key not in matched:
                        matched[key] = (gp_flux[idx], err_list[i])

            # Sort keys and build arrays
            times = np.array(sorted(matched.keys()))
            flux = np.array([matched[float(np.round(t, 4))][0] for t in times])
            errs = np.array([matched[float(np.round(t, 4))][1] for t in times])

            return times, flux, errs


        max_retries = 30

        if spec_tag == 'TDE':
            max_retries = 100

        #threshold_obs = 75 #per 180 days
        threshold_obs = 25 #per 180 days - redefined
        #change to 5 sigma detections  ***
        #WFD cutoff threshold from Sjoert

        #if DDF_tag == True :
            #threshold_obs = 125 # per 180 days
            #threshold_obs = 75 # per 180 days - redefined
        #Redefining parameters for DDF

        best_transient_time_points = 0
        #Setting placeholder best values

        for retry in range(max_retries):
            print(f'Cadence attempt {retry + 1} :')
            sim_date = random.randint(lsst_e_date, lsst_l_date)
            sim_obj_ra = random.randint(0,360)
            sim_obj_dec = random.randint(lsst_lower_dec_limit, lsst_upper_dec_limit)

            #Generating random date within LSST bounds

            #if (retry + 1) == max_retries:
            #    sim_date = best_sim_date
            #    sim_obj_ra = best_sim_obj_ra
            #    sim_obj_dec = best_sim_obj_dec
            #Using best values for final run

            print(sim_date)
            print('RA : ', sim_obj_ra)
            print('Dec : ', sim_obj_dec)
            #Print statements to check values

            baseline_file = get_baseline()
            name = os.path.basename(baseline_file).replace('.db','')
            out_dir = 'temp'
            results_db = maf.db.ResultsDb(out_dir=out_dir)
            bundle_list = []
            metric = maf.metrics.PassMetric(cols=['filter', 'observationStartMJD', 'fiveSigmaDepth', 'visitExposureTime'])
            # Select all the visits. Could be something like "filter='r'", "night < 365", etc

            sql = ''
            slicer = maf.slicers.UserPointsSlicer(ra=sim_obj_ra, dec=sim_obj_dec)
            bundle_list.append(maf.MetricBundle(metric, slicer, sql, run_name=name))
            bd = maf.metricBundles.make_bundles_dict_from_list(bundle_list)
            bg = maf.metricBundles.MetricBundleGroup(bd, baseline_file, out_dir=out_dir, results_db=results_db)
            bg.run_all()
            #Setting up slicer for chosen date, RA and Dec

            try:
                data_slice = bundle_list[0].metric_values[0]
                #Creating data slice

                if isinstance(data_slice, dict):
                    visit_df = pd.DataFrame([data_slice])
                elif isinstance(data_slice, (list, np.ndarray)):
                    visit_df = pd.DataFrame(data_slice)
                else:
                    print(f"[Retry {retry+1}] Unexpected data_slice format: {type(data_slice)} — skipping this run.")
                    continue
                #Checking if data slice has been created in correct format. Skips if not.

                visit_df.sort_values(by='observationStartMJD', inplace=True)
                visit_df = visit_df[['observationStartMJD','filter','fiveSigmaDepth']]
                #Producing list of visits with their respective filters

            except Exception as e:
                print(f"[Retry {retry+1}] Error processing metric values: {e}")
                continue
            #All wrapped in try statement to prevent code from crashing if data slice is incorrectly formatted

            u_visit_list = []; g_visit_list = []; r_visit_list = []; i_visit_list= []; z_visit_list = []; y_visit_list = []
            #Defining lists for visits for each band

            u_ds_mask = visit_df['filter'] == 'u'; g_ds_mask = visit_df['filter'] == 'g'; r_ds_mask = visit_df['filter'] == 'r'
            i_ds_mask = visit_df['filter'] == 'i'; z_ds_mask = visit_df['filter'] == 'z'; y_ds_mask = visit_df['filter'] == 'y'
            #Defining masks for each filter

            u_visit_list = visit_df.loc[u_ds_mask, 'observationStartMJD'].tolist(); g_visit_list = visit_df.loc[g_ds_mask, 'observationStartMJD'].tolist()
            r_visit_list = visit_df.loc[r_ds_mask, 'observationStartMJD'].tolist(); i_visit_list = visit_df.loc[i_ds_mask, 'observationStartMJD'].tolist()
            z_visit_list = visit_df.loc[z_ds_mask, 'observationStartMJD'].tolist(); y_visit_list = visit_df.loc[y_ds_mask, 'observationStartMJD'].tolist()
            #Separating data slice specific to each filter

            print(u_visit_list)

            #all_visit_list_check = np.concatenate([u_visit_list, g_visit_list, r_visit_list, i_visit_list, z_visit_list, y_visit_list])
            #all_visit_list_check = all_visit_list_check[all_visit_list_check>(sim_date-90)]
            #all_visit_list_check = all_visit_list_check[all_visit_list_check<(sim_date+90)]

            #cadence_checkpoint_flag = False

            #if len(all_visit_list_check) < 35:
            #    print(f"Cadence checkpoint failed with only {len(all_visit_list_check)} points within 90 days of peak")
            #    continue

            #print(f"Cadence checkpoint passed with {len(all_visit_list_check)} points within 90 days of peak")
            #cadence_checkpoint_flag = True

            u_visit_err = visit_df.loc[u_ds_mask, 'fiveSigmaDepth'].tolist(); g_visit_err = visit_df.loc[g_ds_mask, 'fiveSigmaDepth'].tolist()
            r_visit_err = visit_df.loc[r_ds_mask, 'fiveSigmaDepth'].tolist(); i_visit_err = visit_df.loc[i_ds_mask, 'fiveSigmaDepth'].tolist()
            z_visit_err = visit_df.loc[z_ds_mask, 'fiveSigmaDepth'].tolist(); y_visit_err = visit_df.loc[y_ds_mask, 'fiveSigmaDepth'].tolist()
            #Separating data slice specific to each filter

            print('u = ',len(u_visit_list)); print('g = ',len(g_visit_list)); print('r = ',len(r_visit_list))
            print('i = ',len(i_visit_list)); print('z = ',len(z_visit_list)); print('y = ',len(y_visit_list))
            #Checking length of each list

            ##### REMOVE LATER TO MAKE SURE YOU DON'T DILATE THE TIME TWICE #####

            #dilation_factor = (1 + lsst_redshift) / (1 + float(redshift))

            #gp_timescale_dilated = gp_timescale * dilation_factor

            gp_timescale_shifted = gp_timescale_dilated + sim_date

            print(f'sim_date_pre_change - {sim_date}')

            #dilated_peak_idx = np.argmax(main_band_flux_gp)
            #dilated_peak_time = gp_timescale_shifted[dilated_peak_idx]
            #sim_date_shift = dilated_peak_time - sim_date
            #print(sim_date_shift)
            #sim_date = sim_date + sim_date_shift
            #print(f'sim_date_post_change - {sim_date}')

            #Applying time dilation

            gp_timescale_shifted_plotting = gp_timescale_shifted - sim_date

            if output_tag == 'test':
                if main_band_tag == 'r':
                    plt.plot(gp_timescale_shifted_plotting,sim_u_flux,label='u',color = filter_colours['u'])
                    plt.plot(gp_timescale_shifted_plotting,sim_g_flux,label='g', color = filter_colours['g'])
                    plt.plot(gp_timescale_shifted_plotting,main_band_flux_gp_scaled,label='r', color = filter_colours['r'])
                    plt.plot(gp_timescale_shifted_plotting,sim_i_flux,label='i', color = filter_colours['i'])
                    plt.plot(gp_timescale_shifted_plotting,sim_z_flux,label='z', color = filter_colours['z'])
                    plt.plot(gp_timescale_shifted_plotting,sim_y_flux,label='y', color = filter_colours['y'])
                    plt.xlabel('Days - Relative to Peak')
                    plt.ylabel('Flux - μJy')
                    #plt.xlim(-750,750)
                    plt.legend()
                    plt.savefig(obj_sub_run_output_path+'/6band_dilated.pdf')
                    plt.close()
                    #plt.show()
                if main_band_tag == 'g':
                    plt.plot(gp_timescale_shifted_plotting,sim_u_flux,label='u', color = filter_colours['u'])
                    plt.plot(gp_timescale_shifted_plotting,main_band_flux_gp_scaled,label='g', color = filter_colours['g'])
                    plt.plot(gp_timescale_shifted_plotting,sim_r_flux,label='r', color = filter_colours['r'])
                    plt.plot(gp_timescale_shifted_plotting,sim_i_flux,label='i', color = filter_colours['i'])
                    plt.plot(gp_timescale_shifted_plotting,sim_z_flux,label='z', color = filter_colours['z'])
                    plt.plot(gp_timescale_shifted_plotting,sim_y_flux,label='y', color = filter_colours['y'])
                    plt.xlabel('Days - Relative to Peak')
                    plt.ylabel('Flux - μJy')
                    plt.legend()
                    plt.savefig(obj_sub_run_output_path+'/6band_dilated.pdf')
                    plt.close()
                    #plt.show()
                #Plotting to check lightcurve

            if main_band_tag == 'r':
                lsst_u_visit_list, lsst_u_visit_flux, lsst_u_visit_err = get_match(u_visit_list, gp_timescale_shifted, sim_u_flux, u_visit_err)
                lsst_g_visit_list, lsst_g_visit_flux, lsst_g_visit_err = get_match(g_visit_list, gp_timescale_shifted, sim_g_flux, g_visit_err)
                lsst_r_visit_list, lsst_r_visit_flux, lsst_r_visit_err = get_match(r_visit_list, gp_timescale_shifted, main_band_flux_gp_scaled, r_visit_err)
                lsst_i_visit_list, lsst_i_visit_flux, lsst_i_visit_err = get_match(i_visit_list, gp_timescale_shifted, sim_i_flux, i_visit_err)
                lsst_z_visit_list, lsst_z_visit_flux, lsst_z_visit_err = get_match(z_visit_list, gp_timescale_shifted, sim_z_flux, z_visit_err)
                lsst_y_visit_list, lsst_y_visit_flux, lsst_y_visit_err = get_match(y_visit_list, gp_timescale_shifted, sim_y_flux, y_visit_err)
            if main_band_tag == 'g':
                lsst_u_visit_list, lsst_u_visit_flux, lsst_u_visit_err = get_match(u_visit_list, gp_timescale_shifted, sim_u_flux, u_visit_err)
                lsst_g_visit_list, lsst_g_visit_flux, lsst_g_visit_err = get_match(g_visit_list, gp_timescale_shifted, main_band_flux_gp_scaled, g_visit_err)
                lsst_r_visit_list, lsst_r_visit_flux, lsst_r_visit_err = get_match(r_visit_list, gp_timescale_shifted, sim_r_flux, r_visit_err)
                lsst_i_visit_list, lsst_i_visit_flux, lsst_i_visit_err = get_match(i_visit_list, gp_timescale_shifted, sim_i_flux, i_visit_err)
                lsst_z_visit_list, lsst_z_visit_flux, lsst_z_visit_err = get_match(z_visit_list, gp_timescale_shifted, sim_z_flux, z_visit_err)
                lsst_y_visit_list, lsst_y_visit_flux, lsst_y_visit_err = get_match(y_visit_list, gp_timescale_shifted, sim_y_flux, y_visit_err)
            #Using function to generate visit dates and matching fluxes for each of the six bands

            if express_tag == True:
                print(f'*Express Mode* - LSST cadence chosen first try')
                break
            #Skipping regeneration if express mode chosen

            all_visit_lists = np.concatenate([lsst_u_visit_list, lsst_g_visit_list, lsst_r_visit_list, lsst_i_visit_list, lsst_z_visit_list, lsst_y_visit_list])
            #all_visit_array = np.array(all_visit_lists)
            #Combine all visit lists into a single array
        
            u_flux_err = ab_to_uJy(lsst_u_visit_err) * 0.2; g_flux_err = ab_to_uJy(lsst_g_visit_err) * 0.2
            r_flux_err = ab_to_uJy(lsst_r_visit_err) * 0.2; i_flux_err = ab_to_uJy(lsst_i_visit_err) * 0.2
            z_flux_err = ab_to_uJy(lsst_z_visit_err) * 0.2; y_flux_err = ab_to_uJy(lsst_y_visit_err) * 0.2
            #Calculating error values for each band

            DDF_tag = False
            if 9.45 - 0.16 < sim_obj_ra < 9.45 + 0.16 and -44 - 0.77 < sim_obj_dec < -44 + 0.77:
                DDF_tag = True
            if 35.7 - 0.62 < sim_obj_ra < 35.7 + 0.62 and -4.75 - 0.08 < sim_obj_dec < -4.75 + 0.08:
                DDF_tag = True
            if 53.2 - 0.93 < sim_obj_ra < 53.2 + 0.93 and -28.1 - 0.49 < sim_obj_dec < -28.1 + 0.49:
                DDF_tag = True
            if 150 - 2.62 < sim_obj_ra < 150 + 2.62 and 2.18 - 0.04 < sim_obj_dec < 2.18 + 0.04:
                DDF_tag = True
            if 58.9 - 1.02 < sim_obj_ra < 58.9 + 1.02 and -49.3 - 0.86 < sim_obj_dec < -49.3 + 0.86:
                DDF_tag = True
            if 63.6 - 1.1 < sim_obj_ra < 63.6 + 1.1 and -47.6 - 0.83 < sim_obj_dec < -47.6 + 0.83:
                DDF_tag = True

            mjd_lsst_u_visit_list = lsst_u_visit_list; mjd_lsst_g_visit_list = lsst_g_visit_list 
            mjd_lsst_r_visit_list = lsst_r_visit_list; mjd_lsst_i_visit_list = lsst_i_visit_list 
            mjd_lsst_z_visit_list = lsst_z_visit_list; mjd_lsst_y_visit_list = lsst_y_visit_list 
            #Saving MJD values for final lightcurve data

            lsst_u_visit_list = lsst_u_visit_list - sim_date; lsst_g_visit_list = lsst_g_visit_list - sim_date 
            lsst_r_visit_list = lsst_r_visit_list - sim_date; lsst_i_visit_list = lsst_i_visit_list - sim_date 
            lsst_z_visit_list = lsst_z_visit_list - sim_date; lsst_y_visit_list = lsst_y_visit_list - sim_date 
            #Converting dates to relative to the peak for plotting

            if output_tag == 'test':
                plt.scatter(lsst_u_visit_list,lsst_u_visit_flux,label='u',color = filter_colours['u'], marker='.')
                plt.scatter(lsst_g_visit_list,lsst_g_visit_flux,label='g', color = filter_colours['g'], marker='.')
                plt.scatter(lsst_r_visit_list,lsst_r_visit_flux,label='r', color = filter_colours['r'], marker='.')
                plt.scatter(lsst_i_visit_list,lsst_i_visit_flux,label='i', color = filter_colours['i'], marker='.')
                plt.scatter(lsst_z_visit_list,lsst_z_visit_flux,label='z', color = filter_colours['z'], marker='.')
                plt.scatter(lsst_y_visit_list,lsst_y_visit_flux,label='y', color = filter_colours['y'], marker='.')
                plt.xlabel('Days - MJD')
                plt.ylabel('Flux - μJy')
                plt.legend()
                #plt.xlim(-750,750)
                plt.savefig(obj_sub_run_output_path+'/6band_cadence.pdf')
                plt.close()
                #plt.show()
                #Plotting six band lsst lightcurve using data points

            ###   Applying Extinction   ###

            coords = SkyCoord(sim_obj_ra,sim_obj_dec, unit='deg')
            SFD = SFDQuery()
            ebv = SFD(coords) * 0.86 #Schafly & Finkbeiner 2011
            #Calculating ebv value usng SFD query

            def flux_extinction (flux, eff_wl):
                A_lambda = fitzpatrick99(eff_wl, ebv * 3.1) #3.1 = Standard Milky Way value
                flux_ext = flux * 10**(-(A_lambda)/2.5)
                return flux_ext
            #Defining function to apply extinction to the flux for a respective band

            u_eff_wl = np.array([3641]); g_eff_wl = np.array([4704]); r_eff_wl = np.array([6155])
            i_eff_wl = np.array([7504]); z_eff_wl = np.array([8695]); y_eff_wl = np.array([10056])
            #Effective wavelength for each band - sourced from SVO Filter Profile Service

            pre_ext_u_flux = lsst_u_visit_flux; pre_ext_g_flux = lsst_g_visit_flux; 
            pre_ext_r_flux = lsst_r_visit_flux; pre_ext_i_flux = lsst_i_visit_flux; 
            pre_ext_z_flux = lsst_z_visit_flux; pre_ext_y_flux = lsst_y_visit_flux; 
            #Saving old values to check extinction is working

            lsst_u_visit_flux = flux_extinction(lsst_u_visit_flux,u_eff_wl)
            lsst_g_visit_flux = flux_extinction(lsst_g_visit_flux,g_eff_wl)
            lsst_r_visit_flux = flux_extinction(lsst_r_visit_flux,r_eff_wl)
            lsst_i_visit_flux = flux_extinction(lsst_i_visit_flux,i_eff_wl)
            lsst_z_visit_flux = flux_extinction(lsst_z_visit_flux,z_eff_wl)
            lsst_y_visit_flux = flux_extinction(lsst_y_visit_flux,y_eff_wl)
            #Applying extinction to the flux from each band

            print('Successfully Extincted Flux')

            cadence_checkpoint_flag = False

            u_f_mask = lsst_u_visit_flux > 5*abs(u_flux_err)
            g_f_mask = lsst_g_visit_flux > 5*abs(g_flux_err)
            r_f_mask = lsst_r_visit_flux > 5*abs(r_flux_err)
            i_f_mask = lsst_i_visit_flux > 5*abs(i_flux_err)
            z_f_mask = lsst_z_visit_flux > 5*abs(z_flux_err)
            y_f_mask = lsst_y_visit_flux > 5*abs(y_flux_err)

            check_u_flux = lsst_u_visit_flux[u_f_mask]; print(len(check_u_flux))
            check_g_flux = lsst_g_visit_flux[g_f_mask]; print(len(check_g_flux))
            check_r_flux = lsst_r_visit_flux[r_f_mask]; print(len(check_r_flux))
            check_i_flux = lsst_i_visit_flux[i_f_mask]; print(len(check_i_flux))
            check_z_flux = lsst_z_visit_flux[z_f_mask]; print(len(check_z_flux))
            check_y_flux = lsst_y_visit_flux[y_f_mask]; print(len(check_y_flux))

            cadence_check_total = len(check_u_flux) + len(check_g_flux) + len(check_r_flux) + len(check_i_flux) + len(check_z_flux) + len(check_y_flux)

            if len(all_visit_lists) == 0:
                print(f"[Retry {retry+1}] No LSST visits found in any band. Skipping.")
                continue

            if cadence_check_total < 10:
                print(f"Cadence checkpoint failed with only {cadence_check_total} 5 sigma detections")
                continue

            cadence_checkpoint_flag = True

            if cadence_checkpoint_flag == True:
                break

        if cadence_checkpoint_flag == False:
            print('Failed to get acceptable cadence. Skipping...')
            fail_reason = 'Failed to achieve acceptable cadence'
            fail_log_entry = pd.DataFrame([{"ZTF Name": object_name[0],
                        "SpecType": SpecType,
                        "Fail Reason": fail_reason}])
            fail_log_entry.to_csv(fail_log_filename, mode='a', index=False, header=False)
            continue

        print(sim_date)
        print('RA : ', sim_obj_ra)
        print('Dec : ', sim_obj_dec)

        if output_tag == 'test':
            plt.scatter(lsst_u_visit_list,lsst_u_visit_flux,label='u',color = filter_colours['u'], marker='.')
            plt.scatter(lsst_g_visit_list,lsst_g_visit_flux,label='g', color = filter_colours['g'], marker='.')
            plt.scatter(lsst_r_visit_list,lsst_r_visit_flux,label='r', color = filter_colours['r'], marker='.')
            plt.scatter(lsst_i_visit_list,lsst_i_visit_flux,label='i', color = filter_colours['i'], marker='.')
            plt.scatter(lsst_z_visit_list,lsst_z_visit_flux,label='z', color = filter_colours['z'], marker='.')
            plt.scatter(lsst_y_visit_list,lsst_y_visit_flux,label='y', color = filter_colours['y'], marker='.')
            plt.xlabel('Days - MJD')
            plt.ylabel('Flux - μJy')
            plt.legend()
            #plt.xlim(-750,750)
            plt.savefig(obj_sub_run_output_path+'/6band_extincted.pdf')
            plt.close()
            #plt.show()
            #Plotting six band lsst lightcurve using data points

        ###  Incorporating error into lightcurve  ###

        #print(f"Average u error = {np.mean(u_flux_err)}. Average g error = {np.mean(g_flux_err)}. Average r error = {np.mean(r_flux_err)}.")
        #print(f"Average i error = {np.mean(i_flux_err)}. Average z error = {np.mean(z_flux_err)}. Average y error = {np.mean(y_flux_err)}.")

        if output_tag == 'test':
            plt.errorbar(lsst_u_visit_list,lsst_u_visit_flux, yerr = u_flux_err,label='u',color = filter_colours['u'], fmt='.', linestyle= None)
            plt.errorbar(lsst_g_visit_list,lsst_g_visit_flux, yerr = g_flux_err,label='g', color = filter_colours['g'], fmt='.', linestyle= None)
            plt.errorbar(lsst_r_visit_list,lsst_r_visit_flux, yerr = r_flux_err,label='r', color = filter_colours['r'], fmt='.', linestyle= None)
            plt.errorbar(lsst_i_visit_list,lsst_i_visit_flux, yerr = i_flux_err,label='i', color = filter_colours['i'], fmt='.', linestyle= None)
            plt.errorbar(lsst_z_visit_list,lsst_z_visit_flux, yerr = z_flux_err,label='z', color = filter_colours['z'], fmt='.', linestyle= None)
            plt.errorbar(lsst_y_visit_list,lsst_y_visit_flux, yerr = y_flux_err,label='y', color = filter_colours['y'], fmt='.', linestyle= None)
            plt.xlabel('Days - MJD')
            plt.ylabel('Flux - μJy')
            plt.legend()
            #plt.xlim(-750,750)
            plt.savefig(obj_sub_run_output_path+'/6band_error.pdf')
            plt.close()
            #plt.show()
            #Plotting points with error bars

        lsst_u_visit_flux = np.random.normal(loc = lsst_u_visit_flux, scale = u_flux_err, size = len(lsst_u_visit_flux))
        lsst_g_visit_flux = np.random.normal(loc = lsst_g_visit_flux, scale = g_flux_err, size = len(lsst_g_visit_flux))
        lsst_r_visit_flux = np.random.normal(loc = lsst_r_visit_flux, scale = r_flux_err, size = len(lsst_r_visit_flux))
        lsst_i_visit_flux = np.random.normal(loc = lsst_i_visit_flux, scale = i_flux_err, size = len(lsst_i_visit_flux))
        lsst_z_visit_flux = np.random.normal(loc = lsst_z_visit_flux, scale = z_flux_err, size = len(lsst_z_visit_flux))
        lsst_y_visit_flux = np.random.normal(loc = lsst_y_visit_flux, scale = y_flux_err, size = len(lsst_y_visit_flux))
        #Applying some random scatter to the lightcurve based on error values

        if output_tag == 'test':
            plt.errorbar(lsst_u_visit_list,lsst_u_visit_flux, yerr = u_flux_err,label='u',color = filter_colours['u'], fmt='.', linestyle= None)
            plt.errorbar(lsst_g_visit_list,lsst_g_visit_flux, yerr = g_flux_err,label='g', color = filter_colours['g'], fmt='.', linestyle= None)
            plt.errorbar(lsst_r_visit_list,lsst_r_visit_flux, yerr = r_flux_err,label='r', color = filter_colours['r'], fmt='.', linestyle= None)
            plt.errorbar(lsst_i_visit_list,lsst_i_visit_flux, yerr = i_flux_err,label='i', color = filter_colours['i'], fmt='.', linestyle= None)
            plt.errorbar(lsst_z_visit_list,lsst_z_visit_flux, yerr = z_flux_err,label='z', color = filter_colours['z'], fmt='.', linestyle= None)
            plt.errorbar(lsst_y_visit_list,lsst_y_visit_flux, yerr = y_flux_err,label='y', color = filter_colours['y'], fmt='.', linestyle= None)
            plt.xlabel('Days - MJD')
            plt.ylabel('Flux - μJy')
            plt.legend()
            #plt.xlim(-750,750)
            plt.savefig(obj_sub_run_output_path+'/6band_final.pdf')
            plt.close()
            #plt.show()
            #Plotting final lightcurve

        print(f'{randir_narthan} - Done!')

        run_end_time = time.time()
        run_time = run_end_time-sub_run_start_time + pre_sub_run_time
        #Determining end time for that run

        #####   Saving output files to new directory   #####

        #with open((obj_sub_run_output_path + "/sim_log.txt"), "w") as file:
        #    file.write(f"Sindarin Name = {randir_narthan}\n")
        #    file.write(f"English Translation = {randir_narthan_eng}\n")
        #    file.write(f"ZTF Object Name = {object_name[0]}\n")
        #    file.write(f"ZTF Object Redshift = {redshift}\n")
        #    file.write(f"ZTF Object Distance = {dist_ZTF:.4g}\n")
        #    file.write(f"Main Band Tag = {main_band_tag}\n")
        #    file.write(f"Express Tag = {express_tag}\n")
        #    file.write(f"DDF Tag = {DDF_tag}\n")
        #    file.write(f"GP Attempts = {main_band_attempt+1}\n")
        #    file.write(f"MSE value = {mse_mb}\n")
        #    file.write(f"Type = {SpecType}\n")
        #    file.write(f"Model chosen = {model_name}\n")
        #    file.write(f"Sim MJD = {sim_date}\n")
        #    file.write(f"Sim RA = {sim_obj_ra}\n")
        #    file.write(f"Sim Dec = {sim_obj_dec}\n")
        #    file.write(f"LSST Redshift = {lsst_redshift:.4g}\n")
        #    file.write(f"LSST Distance = {dist_LSST:.4g}\n")
        #    file.write(f"Run Time = {run_time:.4g}\n")
        #Saving output log file

        if len(lsst_r_visit_flux) > 0:
            lsst_peak_r = np.max(lsst_r_visit_flux)
            lsst_peak_r_time = mjd_lsst_r_visit_list[np.argmax(lsst_r_visit_flux)]
            print(lsst_peak_r_time)
            mag_lsst_peak_r = uJy_to_ab(lsst_peak_r)
        else:
            mag_lsst_peak_r = 'NaN'

        photo_z_err = 'NaN'

        if training_set_tag == False:
            photo_z_err = 0.02 * (1 + lsst_redshift)
            photo_z = np.random.normal(loc = lsst_redshift, scale = photo_z_err)
            lsst_redshift = round(photo_z, 5)
            photo_z_err = round(photo_z_err, 5)
        
        lightcurve = []

        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        visit_lists = [mjd_lsst_u_visit_list, mjd_lsst_g_visit_list, mjd_lsst_r_visit_list, mjd_lsst_i_visit_list, mjd_lsst_z_visit_list, mjd_lsst_y_visit_list]
        flux_lists =  [lsst_u_visit_flux,  lsst_g_visit_flux,  lsst_r_visit_flux,  lsst_i_visit_flux,  lsst_z_visit_flux,  lsst_y_visit_flux]
        flux_err_lists = [u_flux_err, g_flux_err, r_flux_err, i_flux_err, z_flux_err, y_flux_err]
        #Creating relevant lists for output data

        for band, times, fluxes, errs in zip(bands, visit_lists, flux_lists, flux_err_lists):
            for t, f, e in zip(times, fluxes, errs):
                lightcurve.append({'time (MJD)': t, 'flux (μJy)': f, 'flux_err (μJy)': e, 'filter': band})
        #Saving each observation to a lightcurve list

        short_cut_date = sim_date + 10
        lightcurve_df = pd.DataFrame(lightcurve)
        short_lightcurve = lightcurve_df[lightcurve_df['time (MJD)'] < short_cut_date]
        short_lightcurve = short_lightcurve.to_dict(orient='records')

        spec_type_output = SpecType
        ebv_output = round(ebv, 3)

        if training_set_tag == True:
            full_lc_name = train_full_lightcurve_filename
            short_lc_name = train_short_lightcurve_filename
            output_set_tag = 'Training Set'
        if pub_test_tag == True:
            full_lc_name = test_pub_full_lightcurve_filename
            short_lc_name = test_pub_short_lightcurve_filename
            spec_type_output = 'NaN'
            output_set_tag = 'Public Testing Set'
        if priv_test_tag == True:
            full_lc_name = test_priv_full_lightcurve_filename
            short_lc_name = test_priv_short_lightcurve_filename
            spec_type_output = 'NaN'
            output_set_tag = 'Private Testing Set'

        metadata_log_entry = pd.DataFrame([{"Object_ID": randir_narthan,
                                            "Z": f"{float(lsst_redshift):.4g}",
                                        "Z_err": photo_z_err,
                                        "EBV": ebv_output,
                                        "SpecType": spec_type_output,
                                        "English Translation": randir_narthan_eng}])
        
        if training_set_tag == True:
            metadata_log_entry.to_csv(train_log_filename, mode='a', index=False, header=False)
        if pub_test_tag == True:
            metadata_log_entry.to_csv(test_pub_log_filename, mode='a', index=False, header=False)
        if priv_test_tag == True:
            metadata_log_entry.to_csv(test_priv_log_filename, mode='a', index=False, header=False)

        for obs in lightcurve:
            lightcurve_log_entry = pd.DataFrame([{"Object_ID": randir_narthan,
                                    "Time (MJD)": f"{float(obs["time (MJD)"]):.4f}",
                                    "Flux": f"{float(obs['flux (μJy)']):.8f}",
                                    "Flux_err": f"{float(obs['flux_err (μJy)']):.8f}",
                                    "Filter": obs['filter']}])
            lightcurve_log_entry.to_csv(full_lc_name, mode='a', index=False, header=False)

        for obs in short_lightcurve:
            short_lightcurve_log_entry = pd.DataFrame([{"Object_ID": randir_narthan,
                                    "Time (MJD)": f"{float(obs["time (MJD)"]):.4f}",
                                    "Flux": f"{float(obs['flux (μJy)']):.8f}",
                                    "Flux_err": f"{float(obs['flux_err (μJy)']):.8f}",
                                    "Filter": obs['filter']}])
            short_lightcurve_log_entry.to_csv(short_lc_name, mode='a', index=False, header=False)

        master_log_entry = pd.DataFrame([{"Sindarin Name": randir_narthan,
                            "English Translation": randir_narthan_eng,
                            "ZTF Name": object_name[0],
                            "SpecType": SpecType,
                            "Z (ZTF)": redshift ,
                            "Z (LSST)": f"{float(lsst_redshift):.4g}",
                            "Peak Time": sim_date,
                            "LSST r Peak": f"{float(mag_lsst_peak_r):.8f}",
                            "GP Attempts": (main_band_attempt+1),
                            "MSE": mse_mb,
                            "Model": model_name,
                            "Sim RA": sim_obj_ra,
                            "Sim Dec": sim_obj_dec,
                            "Fudge": fudge,
                            "Cadence Attempts": (retry + 1),
                            "5 Sigma Points": (cadence_check_total),
                            "EBV": {ebv_output},
                            "DDF Tag": DDF_tag,
                            "Output Location": output_set_tag,
                            "Run Time": f"{float(run_time):.4g}"}])
        #Writing master log entry for subrun

        master_log_entry.to_csv(master_log_filename, mode='a', index=False, header=False)
        #Appending master log entry to master log

        run_time_list.append(run_time)
        if len(lsst_r_visit_flux) > 0:
            lsst_peak_r = np.max(lsst_r_visit_flux)
            lsst_peak_r_time = mjd_lsst_r_visit_list[np.argmax(lsst_r_visit_flux)]
            print(lsst_peak_r_time)
            mag_lsst_peak_r = uJy_to_ab(lsst_peak_r)
    
            if np.isfinite(mag_lsst_peak_r):
                lsst_peaks_list.append(mag_lsst_peak_r)
        redshift_list.append(lsst_redshift)
        #Appending runtime, redshift and peak mags to relevant lists. Added if statement to prevent crash in case of NaN values.

train_log_df = pd.read_csv(train_log_filename, usecols=obj_log_cols)
train_full_lc_df = pd.read_csv(train_full_lightcurve_filename, usecols=lc_cols)
train_short_lc_df = pd.read_csv(train_short_lightcurve_filename, usecols=lc_cols)

train_log_df = train_log_df.sort_values('Object_ID')
train_full_lc_df = train_full_lc_df.sort_values(['Object_ID', 'Time (MJD)'])
train_short_lc_df = train_short_lc_df.sort_values(['Object_ID', 'Time (MJD)'])

mix_train_log_filename = master_output_path + "train_log.csv"
train_log_df.to_csv(mix_train_log_filename, index=False, columns = obj_log_cols)
mix_train_full_lightcurve_filename = master_output_path + "train_full_lightcurves.csv"
train_full_lc_df.to_csv(mix_train_full_lightcurve_filename, index=False, columns = lc_cols)
mix_train_short_lightcurve_filename = master_output_path + "train_short_lightcurves.csv"
train_short_lc_df.to_csv(mix_train_short_lightcurve_filename, index=False, columns = lc_cols)

test_pub_log_df = pd.read_csv(test_pub_log_filename, usecols=obj_log_cols)
test_pub_full_lc_df = pd.read_csv(test_pub_full_lightcurve_filename, usecols=lc_cols)
test_pub_short_lc_df = pd.read_csv(test_pub_short_lightcurve_filename, usecols=lc_cols)

test_pub_log_df = test_pub_log_df.sort_values('Object_ID')
test_pub_full_lc_df = test_pub_full_lc_df.sort_values(['Object_ID', 'Time (MJD)'])
test_pub_short_lc_df = test_pub_short_lc_df.sort_values(['Object_ID', 'Time (MJD)'])

mix_test_pub_log_filename = master_output_path + "test_pub_log.csv"
test_pub_log_df.to_csv(mix_test_pub_log_filename, index=False, columns = obj_log_cols)
mix_test_pub_full_lightcurve_filename = master_output_path + "test_pub_full_lightcurves.csv"
test_pub_full_lc_df.to_csv(mix_test_pub_full_lightcurve_filename, index=False, columns = lc_cols)
mix_test_pub_short_lightcurve_filename = master_output_path + "test_pub_short_lightcurves.csv"
test_pub_short_lc_df.to_csv(mix_test_pub_short_lightcurve_filename, index=False, columns = lc_cols)

test_priv_log_df = pd.read_csv(test_priv_log_filename, usecols=obj_log_cols)
test_priv_full_lc_df = pd.read_csv(test_priv_full_lightcurve_filename, usecols=lc_cols)
test_priv_short_lc_df = pd.read_csv(test_priv_short_lightcurve_filename, usecols=lc_cols)

test_priv_log_df = test_priv_log_df.sort_values('Object_ID')
test_priv_full_lc_df = test_priv_full_lc_df.sort_values(['Object_ID', 'Time (MJD)'])
test_priv_short_lc_df = test_priv_short_lc_df.sort_values(['Object_ID', 'Time (MJD)'])

mix_test_priv_log_filename = master_output_path + "test_priv_log.csv"
test_priv_log_df.to_csv(mix_test_priv_log_filename, index=False, columns = obj_log_cols)
mix_test_priv_full_lightcurve_filename = master_output_path + "test_priv_full_lightcurves.csv"
test_priv_full_lc_df.to_csv(mix_test_priv_full_lightcurve_filename, index=False, columns = lc_cols)
mix_test_priv_short_lightcurve_filename = master_output_path + "test_priv_short_lightcurves.csv"
test_priv_short_lc_df.to_csv(mix_test_priv_short_lightcurve_filename, index=False, columns = lc_cols)

overall_end_time = time.time()
overall_time = overall_end_time - overall_start_time

print(f"Overall Run Time = {overall_time}")

counts, bins = np.histogram(lsst_peaks_list,bins=10)
counts2, bins2 = np.histogram(run_time_list,bins=10)
counts3, bins3 = np.histogram(redshift_list,bins=10)

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
axes[0].stairs(counts,bins)
axes[0].set_title('LSST Peak Mags')
axes[0].set_xlabel('Mag')
axes[1].stairs(counts2, bins2)
axes[1].set_title('Run Times')
axes[1].set_xlabel('s')
axes[2].stairs(counts3, bins3)
axes[2].set_title('Redshift')
axes[2].set_xlabel('z')
plt.tight_layout()
plt.savefig(master_output_path + '/histograms.pdf')
plt.close()
#plt.show()

def is_valid_number(x):
    try:
        return x is not None and not np.isnan(x)
    except TypeError:
        return False

mse_list = [x for x in mse_list if is_valid_number(x)]
counts4, bins4 = np.histogram(mse_list,10)

fig2, axes2 = plt.subplots(1, 1, figsize=(10, 4), sharey=True)
axes2.stairs(counts4,bins4)
axes2.set_title('MSE Values')
axes2.set_xlabel('MSE Value')
plt.tight_layout()
plt.savefig(master_output_path + '/mse_histogram.pdf')
plt.close()
#plt.show()

lc_counts = {'Ia': Ia_num,'Ia-91T': Ia91T_num,'Ia-91bg': Ia91bg_num,'Ia-02cx': Ia02cx_num,'Ia-02es': Ia02es_num,'Ia-pec': Iapec_num,
    'Ib': Ib_num,'Ibc': Ibc_num,'Ib-pec': Ibpec_num,'Ic': Ic_num,'Ic-BL': IcBL_num,'II': II_num,'IIb': IIb_num,'IIn': IIn_num,
    'SLSN-I': SLSNI_num,'SLSN-II': SLSNII_num,'TDE': TDE_num,'AGN': AGN_num}

plt.figure(figsize=(12, 6))
plt.bar(lc_counts.keys(), lc_counts.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Number of Lightcurves')
plt.tight_layout()
plt.savefig(master_output_path + '/type_chart.pdf')
plt.close()
#plt.show()



