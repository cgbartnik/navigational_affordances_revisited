#### Code for the correlation of spaces with fMRI and EEG brian data

import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore
import cv2
from PIL import Image
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import mne
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn import linear_model
from scipy.stats import ttest_ind, wilcoxon, ttest_rel



EEG_list_action_sorted = ['outdoor_manmade_0147',  'outdoor_manmade_0148',  'outdoor_natural_0246',
 'outdoor_natural_0062',   'outdoor_natural_0160',   'outdoor_natural_0255',
 'outdoor_natural_0128',   'indoor_0156',   'outdoor_manmade_0173',
 'outdoor_manmade_0089',   'outdoor_natural_0104', 'outdoor_natural_0273',
 'outdoor_natural_0079',  'outdoor_manmade_0175',  'outdoor_natural_0042',
 'outdoor_natural_0198',  'outdoor_manmade_0131',  'outdoor_natural_0091',
 'outdoor_manmade_0152',  'outdoor_natural_0200',  'outdoor_manmade_0157',
 'outdoor_manmade_0155',  'indoor_0282',  'outdoor_manmade_0256',
 'outdoor_manmade_0257',  'outdoor_natural_0011',  'indoor_0066',
 'outdoor_manmade_0119',  'outdoor_manmade_0220',  'outdoor_manmade_0068',
 'outdoor_manmade_0133',  'outdoor_manmade_0258',  'outdoor_manmade_0040',
 'outdoor_natural_0132',  'outdoor_manmade_0064',  'outdoor_manmade_0032',
 'outdoor_manmade_0063',  'outdoor_manmade_0015',  'outdoor_manmade_0110',
 'outdoor_manmade_0167',  'outdoor_manmade_0117',  'outdoor_manmade_0030',
 'outdoor_natural_0207',  'outdoor_natural_0053',  'outdoor_natural_0261',
 'outdoor_natural_0097',  'outdoor_natural_0004',  'outdoor_manmade_0149',
 'outdoor_natural_0034',  'outdoor_manmade_0161',  'indoor_0033',
 'indoor_0163',  'indoor_0235',  'indoor_0100',
 'indoor_0058',  'indoor_0145',  'indoor_0271',
 'indoor_0266',  'indoor_0130',  'outdoor_manmade_0276',
 'indoor_0025',  'indoor_0021',  'outdoor_manmade_0165',
 'indoor_0283',  'indoor_0136',  'indoor_0249',
 'indoor_0279',  'indoor_0215',  'indoor_0221',
 'indoor_0216',  'indoor_0214',  'indoor_0080',
 'indoor_0103',  'indoor_0146',  'indoor_0055',
 'indoor_0212',  'indoor_0281',  'outdoor_manmade_0154',
 'indoor_0270',  'outdoor_natural_0049',  'outdoor_natural_0009',
 'outdoor_natural_0010',  'indoor_0272',  'outdoor_natural_0008',
 'outdoor_natural_0052',  'outdoor_natural_0023',  'outdoor_natural_0250',
 'outdoor_natural_0050',  'outdoor_natural_0017',  'outdoor_natural_0252']



def load_and_sort_rdm(rdm_path, ordering, compressed = False):
    fMRI_stim_ordering = ['indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
       'indoor_0058', 'indoor_0066', 'indoor_0080', 'indoor_0100',
       'indoor_0103', 'indoor_0130', 'indoor_0136', 'indoor_0145',
       'indoor_0146', 'indoor_0156', 'indoor_0163', 'indoor_0212',
       'indoor_0214', 'indoor_0215', 'indoor_0216', 'indoor_0221',
       'indoor_0235', 'indoor_0249', 'indoor_0266', 'indoor_0270',
       'indoor_0271', 'indoor_0272', 'indoor_0279', 'indoor_0281',
       'indoor_0282', 'indoor_0283', 'outdoor_manmade_0015',
       'outdoor_manmade_0030', 'outdoor_manmade_0032', 'outdoor_manmade_0040',
       'outdoor_manmade_0063', 'outdoor_manmade_0064', 'outdoor_manmade_0068',
       'outdoor_manmade_0089', 'outdoor_manmade_0110', 'outdoor_manmade_0117',
       'outdoor_manmade_0119', 'outdoor_manmade_0131', 'outdoor_manmade_0133',
       'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_manmade_0149',
       'outdoor_manmade_0152', 'outdoor_manmade_0154', 'outdoor_manmade_0155',
       'outdoor_manmade_0157', 'outdoor_manmade_0161', 'outdoor_manmade_0165',
       'outdoor_manmade_0167', 'outdoor_manmade_0173', 'outdoor_manmade_0175',
       'outdoor_manmade_0220', 'outdoor_manmade_0256', 'outdoor_manmade_0257',
       'outdoor_manmade_0258', 'outdoor_manmade_0276', 'outdoor_natural_0004',
       'outdoor_natural_0008', 'outdoor_natural_0009', 'outdoor_natural_0010',
       'outdoor_natural_0011', 'outdoor_natural_0017', 'outdoor_natural_0023',
       'outdoor_natural_0034', 'outdoor_natural_0042', 'outdoor_natural_0049',
       'outdoor_natural_0050', 'outdoor_natural_0052', 'outdoor_natural_0053',
       'outdoor_natural_0062', 'outdoor_natural_0079', 'outdoor_natural_0091',
       'outdoor_natural_0097', 'outdoor_natural_0104', 'outdoor_natural_0128',
       'outdoor_natural_0132', 'outdoor_natural_0160', 'outdoor_natural_0198',
       'outdoor_natural_0200', 'outdoor_natural_0207', 'outdoor_natural_0246',
       'outdoor_natural_0250', 'outdoor_natural_0252', 'outdoor_natural_0255',
       'outdoor_natural_0261', 'outdoor_natural_0273']
    
    fmri_ordering = [x.replace("_", "") for x in fMRI_stim_ordering]

    if compressed !=False:
        rdm = np.load(rdm_path)["arr_0"]
        if rdm.shape[0] != 90:
                rdm = np.mean(rdm, axis=0)
    else: 
        rdm = np.load(rdm_path)
        if rdm.shape[0] != 90:
            rdm = np.mean(rdm, axis=0)
        
    rdm_df = pd.DataFrame(rdm)
    rdm_df.index = fmri_ordering
    rdm_df.columns = fmri_ordering
    sorted_rdm = rdm_df.loc[ordering, ordering].values
    return sorted_rdm

indoor_images = [img for img in EEG_list_action_sorted if img.startswith('indoor_')]

# Path related

mean_path_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/mean_path_20_20_tiles_euclidean_EEG_sorted.npy")
three_bin_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/three_region_EEG_euc.npy")
angular_8_bin_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_8_EEG_euc.npy")
angular_180_bin_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/angular_bins_180_EEG_euc.npy")

start_point_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/start_point_EEG_euc.npy")
end_point_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/end_point_EEG_euc.npy")
floor_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/floor_20_20_tiles_euclidean_EEG_sorted.npy")
SC_mean_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/rdm_sc_mean_euc_EEG.npy")
CE_mean_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/rdm_ce_mean_euc_EEG.npy")
ce_sc_both = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/rdm_both_ce_sc_euc_EEG.npy")

# Behavior
metric = "euclidean"

action_eeg_rdm = np.load(f"/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/action_average_RDM_{metric}.npy")
object_eeg_rdm = np.load(f"/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/object_average_RDM_{metric}.npy")

action_online_rdm = np.load(f"/home/clemens-uva/Github_repos/EEG/EEG_final/Model_RDMS/online_action_rdm_EGG_action_sorted_{metric}.npy")
object_online_rdm = np.load(f"/home/clemens-uva/Github_repos/EEG/EEG_final/Model_RDMS/online_object_rdm_EGG_action_sorted_{metric}.npy")


images_name = [x.replace("_", "") for x in EEG_list_action_sorted]

fmri_action = load_and_sort_rdm("/home/clemens-uva/Github_repos/Visact_fMRI/code/VISACT_behavior/VISACT_fmri_behavior/fmri_behavior_action_rdms.npz", images_name, compressed=True)


GIST_265 = load_and_sort_rdm('/home/clemens-uva/Github_repos/Visact_fMRI/fMRI_folder/VISACT_RDM_collection/GIST/VISACT_fMRI/GIST_256_RDM_fMRI.npy', images_name)

Gist_euc = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/GIST_265_EEG_euclidean.npy")
Gist_corr = np.load("/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/RDMs/GIST_265_EEG_correlation.npy")




#event dictionary
event_dict = { 'indoor0156' : 4033, 'indoor0282' : 3852, 'indoor0270' : 4064, 'indoor0272' : 4007, 'indoor0066' : 4023, 'indoor0283' : 3898, 'indoor0214' : 3953, 'indoor0080' : 4055, 'indoor0215' : 3964, 'indoor0216': 3931, 'indoor0146' : 4074, 'indoor0221' : 4045, 'indoor0235': 4071, 'indoor0212' : 3960, 'indoor0058' : 4047, 'indoor0145' : 3989, 'indoor0136' : 4018, 'indoor0130' : 4088, 'indoor0163' : 3894, 'indoor0103': 4017,'indoor0100' : 3842, 'indoor0055' : 3858, 'indoor0021' : 3888, 'indoor0266': 3853, 'indoor0025' : 4062, 'indoor0279' : 4027, 'indoor0281' : 3873, 'indoor0271' : 4014, 'indoor0249' : 4002, 'indoor0033' : 4085, 'outdoornatural0010' : 4020, 'outdoornatural0009' : 3981, 'outdoornatural0049' : 3942, 'outdoornatural0008' : 3903, 'outdoornatural0052' : 4076, 'outdoornatural0050' : 4072, 'outdoornatural0132' : 3914, 'outdoornatural0053' : 3930, 'outdoornatural0004' : 3984, 'outdoornatural0207' : 3997, 'outdoornatural0097' : 4003, 'outdoornatural0261' : 4056, 'outdoornatural0011' : 4075, 'outdoornatural0198' : 4063, 'outdoornatural0128' : 3971, 'outdoornatural0255' : 3955, 'outdoornatural0062' : 3925, 'outdoornatural0246' : 3994, 'outdoornatural0160' : 3940, 'outdoornatural0091' : 4030, 'outdoornatural0104' : 4000, 'outdoornatural0200' : 3902, 'outdoornatural0273' : 4043, 'outdoornatural0079' : 3944, 'outdoornatural0042' : 3986, 'outdoornatural0034' : 4061, 'outdoornatural0017' : 3950, 'outdoornatural0023' : 3859, 'outdoornatural0252' : 3870, 'outdoornatural0250' : 3884, 'outdoormanmade0167' : 4059, 'outdoormanmade0040' : 3851, 'outdoormanmade0110' : 3841, 'outdoormanmade0117' : 4077, 'outdoormanmade0030': 3891, 'outdoormanmade0258' : 4081, 'outdoormanmade0064' : 3926, 'outdoormanmade0068' : 4038, 'outdoormanmade0063' : 3845, 'outdoormanmade0015' : 3871, 'outdoormanmade0257': 4078, 'outdoormanmade0032' : 3878, 'outdoormanmade0256': 3918, 'outdoormanmade0220' : 4052, 'outdoormanmade0133' : 4013, 'outdoormanmade0119' : 3886, 'outdoormanmade0152' : 4001, 'outdoormanmade0148' : 4083, 'outdoormanmade0155' : 3899, 'outdoormanmade0157' : 3843, 'outdoormanmade0175' : 4048, 'outdoormanmade0173': 3907, 'outdoormanmade0089' : 3862, 'outdoormanmade0147': 4060, 'outdoormanmade0131' : 3874, 'outdoormanmade0161' : 3869, 'outdoormanmade0154' : 4041, 'outdoormanmade0165' : 3854, 'outdoormanmade0276': 3976, 'outdoormanmade0149' : 3866}
#images_name = [ 'indoor0156' , 'indoor0282', 'indoor0270' , 'indoor0272', 'indoor0066', 'indoor0283', 'indoor0214', 'indoor0080' , 'indoor0215', 'indoor0216' , 'indoor0146', 'indoor0221' , 'indoor0235', 'indoor0212' , 'indoor0058' , 'indoor0145', 'indoor0136', 'indoor0130' , 'indoor0163', 'indoor0103','indoor0100' , 'indoor0055', 'indoor0021', 'indoor0266', 'indoor0025', 'indoor0279' , 'indoor0281', 'indoor0271', 'indoor0249' , 'indoor0033', 'outdoornatural0010', 'outdoornatural0009', 'outdoornatural0049' , 'outdoornatural0008' , 'outdoornatural0052' , 'outdoornatural0050' , 'outdoornatural0132' , 'outdoornatural0053' , 'outdoornatural0004' , 'outdoornatural0207', 'outdoornatural0097', 'outdoornatural0261', 'outdoornatural0011' , 'outdoornatural0198' , 'outdoornatural0128' , 'outdoornatural0255', 'outdoornatural0062' , 'outdoornatural0246' , 'outdoornatural0160', 'outdoornatural0091' , 'outdoornatural0104' , 'outdoornatural0200' , 'outdoornatural0273' , 'outdoornatural0079', 'outdoornatural0042' , 'outdoornatural0034' , 'outdoornatural0017', 'outdoornatural0023' , 'outdoornatural0252', 'outdoornatural0250' , 'outdoormanmade0167' , 'outdoormanmade0040' , 'outdoormanmade0110' , 'outdoormanmade0117' , 'outdoormanmade0030', 'outdoormanmade0258' , 'outdoormanmade0064' , 'outdoormanmade0068' , 'outdoormanmade0063', 'outdoormanmade0015' , 'outdoormanmade0257', 'outdoormanmade0032' , 'outdoormanmade0256' , 'outdoormanmade0220'  , 'outdoormanmade0133' , 'outdoormanmade0119' , 'outdoormanmade0152' , 'outdoormanmade0148' , 'outdoormanmade0155', 'outdoormanmade0157', 'outdoormanmade0175' , 'outdoormanmade0173', 'outdoormanmade0089' , 'outdoormanmade0147', 'outdoormanmade0131', 'outdoormanmade0161', 'outdoormanmade0154' , 'outdoormanmade0165' , 'outdoormanmade0276', 'outdoormanmade0149' ]
#event_dict = { 'indoor_0156' : 4033, 'indoor_0282' : 3852, 'indoor_0270' : 4064, 'indoor_0272' : 4007, 'indoor_0066' : 4023, 'indoor_0283' : 3898, 'indoor_0214' : 3953, 'indoor_0080' : 4055, 'indoor_0215' : 3964, 'indoor_0216': 3931, 'indoor_0146' : 4074, 'indoor_0221' : 4045, 'indoor_0235': 4071, 'indoor_0212' : 3960, 'indoor_0058' : 4047, 'indoor_0145' : 3989, 'indoor_0136' : 4018, 'indoor_0130' : 4088, 'indoor_0163' : 3894, 'indoor_0103': 4017,'indoor_0100' : 3842, 'indoor_0055' : 3858, 'indoor_0021' : 3888, 'indoor_0266': 3853, 'indoor_0025' : 4062, 'indoor_0279' : 4027, 'indoor_0281' : 3873, 'indoor_0271' : 4014, 'indoor_0249' : 4002, 'indoor_0033' : 4085, 'outdoor_natural_0010' : 4020, 'outdoor_natural_0009' : 3981, 'outdoor_natural_0049' : 3942, 'outdoor_natural_0008' : 3903, 'outdoor_natural_0052' : 4076, 'outdoor_natural_0050' : 4072, 'outdoor_natural_0132' : 3914, 'outdoor_natural_0053' : 3930, 'outdoor_natural_0004' : 3984, 'outdoor_natural_0207' : 3997, 'outdoor_natural_0097' : 4003, 'outdoor_natural_0261' : 4056, 'outdoor_natural_0011' : 4075, 'outdoor_natural_0198' : 4063, 'outdoor_natural_0128' : 3971, 'outdoor_natural_0255' : 3955, 'outdoor_natural_0062' : 3925, 'outdoor_natural_0246' : 3994, 'outdoor_natural_0160' : 3940, 'outdoor_natural_0091' : 4030, 'outdoor_natural_0104' : 4000, 'outdoor_natural_0200' : 3902, 'outdoor_natural_0273' : 4043, 'outdoor_natural_0079' : 3944, 'outdoor_natural_0042' : 3986, 'outdoor_natural_0034' : 4061, 'outdoor_natural_0017' : 3950, 'outdoor_natural_0023' : 3859, 'outdoor_natural_0252' : 3870, 'outdoor_natural_0250' : 3884, 'outdoor_manmade_0167' : 4059, 'outdoor_manmade_0040' : 3851, 'outdoor_manmade_0110' : 3841, 'outdoor_manmade_0117' : 4077, 'outdoor_manmade_0030': 3891, 'outdoor_manmade_0258' : 4081, 'outdoor_manmade_0064' : 3926, 'outdoor_manmade_0068' : 4038, 'outdoor_manmade_0063' : 3845, 'outdoor_manmade_0015' : 3871, 'outdoor_manmade_0257': 4078, 'outdoor_manmade_0032' : 3878, 'outdoor_manmade_0256': 3918, 'outdoor_manmade_0220' : 4052, 'outdoor_manmade_0133' : 4013, 'outdoor_manmade_0119' : 3886, 'outdoor_manmade_0152' : 4001, 'outdoor_manmade_0148' : 4083, 'outdoor_manmade_0155' : 3899, 'outdoor_manmade_0157' : 3843, 'outdoor_manmade_0175' : 4048, 'outdoor_manmade_0173': 3907, 'outdoor_manmade_0089' : 3862, 'outdoor_manmade_0147': 4060, 'outdoor_manmade_0131' : 3874, 'outdoor_manmade_0161' : 3869, 'outdoor_manmade_0154' : 4041, 'outdoor_manmade_0165' : 3854, 'outdoor_manmade_0276': 3976, 'outdoor_manmade_0149' : 3866}


event_dict = { 'indoor0156' : 4033, 'indoor0282' : 3852, 'indoor0270' : 4064, 'indoor0272' : 4007, 'indoor0066' : 4023, 'indoor0283' : 3898, 'indoor0214' : 3953, 'indoor0080' : 4055, 'indoor0215' : 3964, 'indoor0216': 3931, 'indoor0146' : 4074, 'indoor0221' : 4045, 'indoor0235': 4071, 'indoor0212' : 3960, 'indoor0058' : 4047, 'indoor0145' : 3989, 'indoor0136' : 4018, 'indoor0130' : 4088, 'indoor0163' : 3894, 'indoor0103': 4017,'indoor0100' : 3842, 'indoor0055' : 3858, 'indoor0021' : 3888, 'indoor0266': 3853, 'indoor0025' : 4062, 'indoor0279' : 4027, 'indoor0281' : 3873, 'indoor0271' : 4014, 'indoor0249' : 4002, 'indoor0033' : 4085, 'outdoornatural0010' : 4020, 'outdoornatural0009' : 3981, 'outdoornatural0049' : 3942, 'outdoornatural0008' : 3903, 'outdoornatural0052' : 4076, 'outdoornatural0050' : 4072, 'outdoornatural0132' : 3914, 'outdoornatural0053' : 3930, 'outdoornatural0004' : 3984, 'outdoornatural0207' : 3997, 'outdoornatural0097' : 4003, 'outdoornatural0261' : 4056, 'outdoornatural0011' : 4075, 'outdoornatural0198' : 4063, 'outdoornatural0128' : 3971, 'outdoornatural0255' : 3955, 'outdoornatural0062' : 3925, 'outdoornatural0246' : 3994, 'outdoornatural0160' : 3940, 'outdoornatural0091' : 4030, 'outdoornatural0104' : 4000, 'outdoornatural0200' : 3902, 'outdoornatural0273' : 4043, 'outdoornatural0079' : 3944, 'outdoornatural0042' : 3986, 'outdoornatural0034' : 4061, 'outdoornatural0017' : 3950, 'outdoornatural0023' : 3859, 'outdoornatural0252' : 3870, 'outdoornatural0250' : 3884, 'outdoormanmade0167' : 4059, 'outdoormanmade0040' : 3851, 'outdoormanmade0110' : 3841, 'outdoormanmade0117' : 4077, 'outdoormanmade0030': 3891, 'outdoormanmade0258' : 4081, 'outdoormanmade0064' : 3926, 'outdoormanmade0068' : 4038, 'outdoormanmade0063' : 3845, 'outdoormanmade0015' : 3871, 'outdoormanmade0257': 4078, 'outdoormanmade0032' : 3878, 'outdoormanmade0256': 3918, 'outdoormanmade0220' : 4052, 'outdoormanmade0133' : 4013, 'outdoormanmade0119' : 3886, 'outdoormanmade0152' : 4001, 'outdoormanmade0148' : 4083, 'outdoormanmade0155' : 3899, 'outdoormanmade0157' : 3843, 'outdoormanmade0175' : 4048, 'outdoormanmade0173': 3907, 'outdoormanmade0089' : 3862, 'outdoormanmade0147': 4060, 'outdoormanmade0131' : 3874, 'outdoormanmade0161' : 3869, 'outdoormanmade0154' : 4041, 'outdoormanmade0165' : 3854, 'outdoormanmade0276': 3976, 'outdoormanmade0149' : 3866}
images_name = [x.replace("_", "") for x in EEG_list_action_sorted]


participants_list = ['sapaj', 'ppnjn', 'azrfp', 'cuvfl', 'domdz', 'npcrj', 'hoxev','kuupm',
                    'rxsrg', 'pflzs', 'kktpp', 'pyyor', 'liirj', 'qmrlx', 'jpdoy', 'hapql', 'ghldo', 'fgljq'] # pwixa


#####
# Path to preprocessed data 
DATA_path = "/home/clemens-uva/Github_repos/EEG/EEG_final/DATA/75_downsample128/"

file_substring = sorted(os.listdir(DATA_path))[0]

# settings
tmin                = -0.1
tmax                = 1.0
down_sample_rate    = 128 # might need to be adapted depending on the preprocessing

# Calculate the duration
duration = tmax - tmin

# Calculate the number of time points
n_timepoints = int(duration * down_sample_rate) + 1  # +1 to include both endpoints

# Generate downsampled time points
t = np.linspace(tmin, tmax, n_timepoints)

# channel info
n_channels      = 64

# Epochs
n_epochs        = 540


# sets of electrodes
occipital_electrodes = ['P1',  'P3',  'P5',  'P7',  'P9',  'PO7',  'PO3',  'O1', 'Oz',
 'POz',  'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
frontal_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8']

all_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P1',  'P3',  'P5',  'P7',  'P9',  'PO7',  'PO3',  'O1', 'Oz',
 'POz',  'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']




def corr_with_model(rdm1, model_rdm):
    corrs = []
    for timepoint in range(rdm1.shape[0]):
        rdv1 = squareform(rdm1[timepoint].round(10))
        rdv2 = squareform(model_rdm.round(10))
        corr, p = spearmanr(rdv1, rdv2)
        corrs.append(corr)

    mean = np.mean(corrs)

    return mean, corrs


def compute_corrs_sliding(distance_metric, n, model_rdm):
    path = "/home/clemens-uva/Desktop/EEG---temporal-dynamics-of-affordance-perception/RDMs/ERP_sliding_window_RDMs/"

    all_sub_corrs = []
    for file in os.listdir(path):
        if (distance_metric in file) and (n in file):
            rdms_per_subject = np.load(path+file)
            mean_corr, corrs = corr_with_model(rdms_per_subject, model_rdm)
            all_sub_corrs.append(corrs)
    
    mean_corr = np.mean(np.array(all_sub_corrs), axis = 0)
    sem = np.std(all_sub_corrs, axis=0) / np.sqrt(len(all_sub_corrs))

    return mean_corr, sem, np.array(all_sub_corrs)

''' 
def significant_against_zero(array):
    
    t_values, p_values = ttest_1samp(array, 0, axis=0)
    # Adjust p-values for FDR using Benjamini-Hochberg procedure
    alpha = 0.05
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    # Output the results
    significant_timepoints = np.where(reject)[0]

    
    return significant_timepoints

'''
def significant_against_zero(array):
    t_values, p_values = ttest_1samp(array, 0, axis=0)
    
    # Adjust p-values for FDR using Benjamini-Hochberg procedure
    alpha = 0.05
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Get indices of significant time points
    significant_timepoints = np.where(reject)[0]

    # Extract t-values and p-values for significant time points
    significant_t_values = t_values[significant_timepoints]
    significant_p_values = p_values[significant_timepoints]

    # Print results only for significant time points
    for i, idx in enumerate(significant_timepoints):
        print(f"Timepoint {t[idx]:.3f}: t = {significant_t_values[i]:.2f}, p = {significant_p_values[i]:.3f}")

    return significant_timepoints


def lowest_value(array1, array2):
    """
    Finds the lowest value between two arrays and returns it.

    Parameters:
    - array1: First numpy array.
    - array2: Second numpy array.

    Returns:
    - The lowest value found in both arrays.
    """

    # Find the minimum of each array
    min1 = np.min(array1)
    min2 = np.min(array2)
    
    # Return the minimum of both values
    return min(min1, min2)




def subset_rdm(rdm: np.ndarray, original_image_list: list, new_image_list: list) -> np.ndarray:
    """
    Subset an RDM to include only entries corresponding to the new image list.

    Parameters:
    - rdm: 2D numpy array (RDM), assumed to be square and ordered by original_image_list
    - original_image_list: list of image names used to sort the RDM
    - new_image_list: list of image names to subset from the RDM

    Returns:
    - subsetted_rdm: 2D numpy array with only rows and columns from new_image_list
    """
    # Get indices in the original image list that match the new image list
    indices = [original_image_list.index(img) for img in new_image_list if img in original_image_list]

    # Use numpy indexing to subset the RDM
    subsetted_rdm = rdm[np.ix_(indices, indices)]
    
    return subsetted_rdm



def corr_with_model_subset(rdm1, model_rdm, subset_image_list):

    model_subset = subset_rdm(model_rdm, EEG_list_action_sorted, subset_image_list)
    corrs = []
    for timepoint in range(rdm1.shape[0]):
        timepoint_subset = subset_rdm(rdm1[timepoint], EEG_list_action_sorted, subset_image_list)
        rdv1 = squareform(timepoint_subset.round(10))
        rdv2 = squareform(model_subset.round(10))
        corr, p = spearmanr(rdv1, rdv2)
        corrs.append(corr)

    mean = np.mean(corrs)

    return mean, corrs



def compute_corrs_sliding_subset(distance_metric, n, model_rdm, subset_image_list):
    path = "/home/clemens-uva/Desktop/EEG---temporal-dynamics-of-affordance-perception/RDMs/ERP_sliding_window_RDMs/"

    all_sub_corrs = []
    for file in os.listdir(path):
        if (distance_metric in file) and (n in file):
            rdms_per_subject = np.load(path+file)
            mean_corr, corrs = corr_with_model_subset(rdms_per_subject, model_rdm, subset_image_list)
            all_sub_corrs.append(corrs)
    
    mean_corr = np.mean(np.array(all_sub_corrs), axis = 0)
    sem = np.std(all_sub_corrs, axis=0) / np.sqrt(len(all_sub_corrs))

    return mean_corr, sem, np.array(all_sub_corrs)



def plot_correlation_comparison_subset(spaces, colors, names, subset_image_list, save_path, ref_peaks=True):
    import matplotlib.pyplot as plt
    import numpy as np

    distance_metric = "euclidean"
    line_style = "-"
    alpha_line = 1
    alpha_shades = 0.1
    lw = 3

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(x=0, color='lightgray', linestyle='--')
    ax.axhline(y=0, color='lightgray', linestyle='--')

    all_means = []
    all_sems = []
    all_arrays = []

    min_values = []

    for space, color, name in zip(spaces, colors, names):
        mean, sem, array = compute_corrs_sliding_subset(distance_metric, "_5_", space, subset_image_list)
        all_means.append(mean)
        all_sems.append(sem)
        all_arrays.append(array)

        plt.plot(t, mean, color=color, label=f"{name}", alpha=alpha_line, lw=lw)
        plt.fill_between(t, mean - sem, mean + sem, alpha=alpha_shades, color=color)

        min_values.append(np.max(mean + sem))

    global_min = np.max(min_values)

    for i, (mean, sem, array, color, space_name) in enumerate(zip(all_means, all_sems, all_arrays, colors, names)):
        print(space_name + ":")
        sig = significant_against_zero(array)
        for timepoint in sig:
            plt.text(t[timepoint], global_min + 0.008 * i, color=color, s=".", fontsize=30)

        # plot peak timepoints
        ax.axvline(x=t[np.argmax(np.abs(mean))], color=color, linestyle='--', lw=2)

    if ref_peaks:
        mean, sem, array = compute_corrs_sliding(distance_metric, "_5_", action_eeg_rdm)
        ax.axvline(x=t[np.argmax(np.abs(mean))], color="#ff2c55", linestyle='--', lw=2)
        mean, sem, array = compute_corrs_sliding(distance_metric, "_5_", GIST_265)
        ax.axvline(x=t[np.argmax(np.abs(mean))], color="#ee9b00", linestyle='--', lw=2)



    plt.ylim(-0.15, 0.25)
    plt.xlabel('Time (s)', fontsize=20)
    plt.ylabel('RSA correlation (ρ)', fontsize=18)
    #plt.legend(fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.02, transparent = True)


plot_correlation_comparison_subset(spaces = [action_eeg_rdm, fmri_action, Gist_euc],
                            colors = ["#ff2c55","#f54f8e", "#ee9b00"],
                            names = ["Loc. Aff (EGG)", "Loc. Aff (fMRI)", "GIST"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/Affordance_all_Gist_euc_all_images.png",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [action_eeg_rdm, fmri_action, Gist_euc],
                            colors = ["#ff2c55","#f54f8e", "#ee9b00"],
                            names = ["Loc. Aff (EGG)", "Loc. Aff (fMRI)", "GIST"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/Affordance_all_Gist_euc_indoor.png",
                            ref_peaks=False)

'''
plot_correlation_comparison_subset(spaces = [action_eeg_rdm, Gist_euc],
                            colors = ["#ff2c55", "#ee9b00"],
                            names = ["Loc. Aff","GIST"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/Affordance_Gist_euc_all_images.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [action_eeg_rdm, Gist_euc],
                            colors = ["#ff2c55", "#ee9b00"],
                            names = ["Loc. Aff","GIST"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/Affordance_Gist_euc_all_images.png",
                            ref_peaks=False)


plot_correlation_comparison_subset(spaces = [action_eeg_rdm, Gist_euc],
                            colors = ["#ff2c55", "#ee9b00"],
                            names = ["Loc. Aff","GIST"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/Affordance_Gist_euc_indoor.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [action_eeg_rdm, Gist_euc],
                            colors = ["#ff2c55", "#ee9b00"],
                            names = ["Loc. Aff","GIST"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/Affordance_Gist_euc_indoor.png",
                            ref_peaks=False)






plot_correlation_comparison_subset(spaces = [start_point_euc, end_point_euc],
                             colors = ["#03c042","#7e1db6"],
                            names = [ "Start Point", " End Point"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/start_goal_euc_all_images.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [start_point_euc, end_point_euc],
                            colors = ["#03c042","#7e1db6"],
                            names = [ "Start Point", " End Point"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/start_goal_euc_all_images.png",
                            ref_peaks=False)




plot_correlation_comparison_subset(spaces = [start_point_euc, end_point_euc],
                             colors = ["#03c042","#7e1db6"],
                            names = [ "Start Point", " End Point"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/start_goal_euc_indoor.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [start_point_euc, end_point_euc],
                            colors = ["#03c042","#7e1db6"],
                            names = [ "Start Point", " End Point"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/start_goal_euc_indoor.png",
                            ref_peaks=False)


plot_correlation_comparison_subset(spaces = [start_point_euc, end_point_euc],
                             colors = ["#03c042","#7e1db6"],
                            names = [ "Start Point", " End Point"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/start_goal_euc_indoor.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [start_point_euc, end_point_euc],
                            colors = ["#03c042","#7e1db6"],
                            names = [ "Start Point", " End Point"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/start_goal_euc_indoor.png",
                            ref_peaks=False)




plot_correlation_comparison_subset(spaces = [mean_path_euc, three_bin_euc,  angular_8_bin_euc, angular_180_bin_euc],
                             colors = ['#052f77',  "#7b91b8", "#0dc0d4", "#38fff5" ],
                            names = ["Mean Path", "3 Bins", "8 Bins", "180 Bins"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/mean_path_euc_all_images.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [mean_path_euc, three_bin_euc,  angular_8_bin_euc, angular_180_bin_euc],
                            colors = ['#052f77',  "#7b91b8", "#0dc0d4", "#38fff5" ],
                            names = ["Mean Path", "3 Bins", "8 Bins", "180 Bins"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/mean_path_euc_all_images.png",
                            ref_peaks=False)


plot_correlation_comparison_subset(spaces = [mean_path_euc, three_bin_euc,  angular_8_bin_euc, angular_180_bin_euc],
                             colors = ['#052f77',  "#7b91b8", "#0dc0d4", "#38fff5" ],
                            names = ["Mean Path", "3 Bins", "8 Bins", "180 Bins"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/mean_path_euc_indoor.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [mean_path_euc, three_bin_euc,  angular_8_bin_euc, angular_180_bin_euc],
                            colors = ['#052f77',  "#7b91b8", "#0dc0d4", "#38fff5" ],
                            names = ["Mean Path", "3 Bins", "8 Bins", "180 Bins"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/mean_path_euc_indoor.png",
                            ref_peaks=False)



plot_correlation_comparison_subset(spaces = [floor_euc],
                            colors = ["#42078f"],
                            names = ["Floor"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/floor_euc_all_images.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [floor_euc],
                            colors = ["#42078f"],
                            names = ["Floor"],
                            subset_image_list=EEG_list_action_sorted,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/floor_euc_all_images.png",
                            ref_peaks=False)


plot_correlation_comparison_subset(spaces = [floor_euc],
                            colors = ["#42078f"],
                            names = ["Floor"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/floor_euc_indoor.svg",
                            ref_peaks=False)

plot_correlation_comparison_subset(spaces = [floor_euc],
                            colors = ["#42078f"],
                            names = ["Floor"],
                            subset_image_list=indoor_images,
                            save_path = "/home/clemens-uva/Desktop/XAI---locomotive-affordance-perception/Figures/floor_euc_indoor.png",
                            ref_peaks=False)

                            '''