import os 
import sys
import numpy as np
import pandas as pd
import astropy.units as u


class lasair_objects:

    '''
    This class is used to be consistant with Lasair-LSST objects Table schema.
    '''
    def __init__(self, object_df):
        self.diaObjectId = object_df['Object_ID']
        self.ra, self.decl = None, None 
        self.lastDiaSourceMjdTai = None
        self.firstDiaSourceMjdTai = None
        

    

    