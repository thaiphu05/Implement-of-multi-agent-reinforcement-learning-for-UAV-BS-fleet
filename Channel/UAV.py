import numpy as np
import matplotlib.pyplot as plt
import math
import os

class UAV:
    x = None
    y = None
    def __init__(self, height, velocity, p_tx_uav_dbm):
        self.h = height
        self.velocity = velocity
        self.p_tx_uav_dbm = p_tx_uav_dbm

    def set_position_2D(self, x, y):
        self.x = x
        self.y = y

    def distance_to_user(self, user_x, user_y):
        return np.sqrt((self.x - user_x)**2 + (self.y - user_y)**2)