import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn



def get_mallorn_data(data_path):
    data = pd.read_csv(data_path)
    return data





if __name__ == "__main__":
    data = get_mallorn_data("data/mallorn.csv")
    print(data.head())