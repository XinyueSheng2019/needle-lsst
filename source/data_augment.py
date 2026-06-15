'''
==============================================
data_augment.py
==============================================

This file contains the functions to augment the data for the model.
'''
import random 
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DataAugmentation(nn.Module.Linear):
    '''
    data agumentation layer for the needle model.
    '''

    def __init__(self, version = 'raw' | 'benchmark' | 'advanced' | 'extracted'):
        self.version = version

    def get_augmented_data(self, data):
        if self.version == 'advanced':
            return self.get_advanced_data(data)
        elif self.version == 'benchmark':
            return self.get_benchmark_data(data)
        elif self.version == 'raw':
            return self.get_raw_data(data)
        elif self.version == 'extracted':
            return self.get_extracted_data(data)
        else:
            raise ValueError(f"Invalid version: {self.version}, please choose from 'raw', 'benchmark', 'advanced', or 'extracted'")

    def get_benchmark_data(self, data):
        return data

    def get_advanced_data(self, data):
        return data

    def get_raw_data(self, data):
        return data

    def get_extracted_data(self, data):
        return data
