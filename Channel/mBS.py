import numpy as np
import matplotlib.pyplot as plt
import math
import os

class mBS:
    x = None
    y = None
    def __init__(self, height, p_tx_mbs_dbm):
        self.h = height
        self.p_tx_mBS_dbm = p_tx_mbs_dbm

    def set_position_2D(self, x, y):
        self.x = x
        self.y = y

    def distance_to_user(self, user_x, user_y):
        return np.sqrt((self.x - user_x)**2 + (self.y - user_y)**2)