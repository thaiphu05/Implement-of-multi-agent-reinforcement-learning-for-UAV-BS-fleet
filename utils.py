import numpy as np
import math
import matplotlib.pyplot as plt
import os

def calculate_rate(snr, W_bandwidth):
    """Calculate the data rate based on the SNR using Shannon - Hartley formula."""
    return W_bandwidth * np.log2(1 + snr)

def indicator(r_threshold, rate):
    """Indicator function that returns 1 if the rate meets the threshold, else 0."""
    return 1 if rate >= r_threshold else 0



