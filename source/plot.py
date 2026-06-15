import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch



def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.show()



if __name__ == "__main__":
    loss_history = [0.5, 0.4, 0.3, 0.2, 0.1]
    plot_loss(loss_history)