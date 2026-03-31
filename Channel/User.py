import numpy as np
import matplotlib.pyplot as plt
import math
import os

class User:
    def __init__(self, x, y, rate_threshold):
        self.x = x
        self.y = y
        self.rate_threshold = rate_threshold
        self.connected = False
