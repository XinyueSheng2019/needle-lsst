import random 
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DataAugmentation:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def augment_data(self):
        pass

    def augment_labels(self):
        pass