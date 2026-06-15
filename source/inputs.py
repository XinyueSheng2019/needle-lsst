'''
==============================================
inputs.py
==============================================

This file contains the functions to get the data from the Mallorn dataset.
'''
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MallornData:
    def __init__(self, data_path, log_path):
        self.data = pd.read_csv(data_path)

    def get_data(self):
        return self.data

    def get_log(self):
        return self.log
    
    def get_discovery(self, sigma_limit = 5):
        # return self.data[self.data['Z'] > sigma_limit]
        '''
        this function will return the data that is within the sigma limit of the discovery.
        return: [discovery_time, discovery_flux, discovery_flux_error, discovery_filter]
        '''
        return pass 

    def remove_outliers(self, sigma_limit = 5):
        '''
        this function will remove the outliers from the data.
        return: [time, flux, flux_error, filter]
        '''
        return pass 
    
    def save_to_npy(self, data, path):
        '''
        this function will save the data to a file.
        return: None
        '''
        return pass 

    def load_from_npy(self, path):
        '''
        this function will load the data from a file.
        return: [time, flux, flux_error, filter]
        '''
        return pass 


    @property
    def data(self):
        return self.get_data()
    
    @property
    def log(self):
        return self.get_log()

    @property
    def discovery(self):
        return self.get_discovery()





