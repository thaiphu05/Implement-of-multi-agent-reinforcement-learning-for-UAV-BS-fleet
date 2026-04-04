import numpy as np 
import matplotlib.pyplot as plt
import math
import os 

class Channel_Model_UAV:
    def __init__(self, f_c=2e9, alpha=2.5, sigma2_dbm=-110):
            """
            :param f_c: Carrier frequency (Hz)
            :param alpha: path-loss exponent
            :param sigma2_dbm: Noise power (dBm)
            """
            self.f_c = f_c
            self.alpha = alpha
            self.c = 3e8 
            self.lamda = self.c / self.f_c
            self.d_ref = 1
                        
            self.sigma2 = 10**((sigma2_dbm - 30) / 10)
    def get_path_loss_gain(self, d_2D, h_UAV, fading_power=None):
        """
        Calculate the path loss gain based on the distance between the UAV and the user.
        :param d_2D: Distance between UAV and user (m)
        :param h_UAV: UAV height (m)
        :return: Path loss gain
        """
        d_3D = np.sqrt(d_2D**2 + h_UAV**2)
        
        if fading_power is None:
            fading_power = np.random.rayleigh(scale=1.0) ** 2
        
        theta_db = -20 * np.log10(4 * np.pi * self.d_ref / self.lamda)
        theta_linear = 10 ** (theta_db / 10)
        path_loss_gain = fading_power * theta_linear * ((d_3D/ self.d_ref ) ** (-self.alpha))
        
        return path_loss_gain
    def get_snr(self, d_2D, h_UAV, p_tx_dbm, fading_power=None):
        """
        Calculate the SNR based on the path loss gain and noise power.
        :param d_2D: Distance between UAV and user (m)
        :param h_UAV: UAV height (m)
        :param p_tx_dbm: Transmit power (dBm)
        :return: SNR
        """
        path_loss_gain = self.get_path_loss_gain(d_2D, h_UAV, fading_power=fading_power)
        p_tx_w = 10**((p_tx_dbm - 30) / 10)
        snr_linear = p_tx_w * path_loss_gain / self.sigma2
        return snr_linear

class Channel_Model_mBS:
    def __init__(self, f_c=2e9, sigma2_dbm=-110, d_ref=1):
        self.f_c = f_c
        self.sigma2_dbm = sigma2_dbm
        self.d_ref = d_ref
        self.c = 3e8
    
    def path_loss_db(self, d_2D, h_mBS):
        """
        Calculate path loss in dB based on the distance.
        :param d_2D: 2D distance between mBS and user (m)
        :param h_mBS: Height of the mBS (m)
        :return: Path loss in dB
        """
        d_3D = np.sqrt(d_2D**2 + h_mBS**2)/1000
        # The empirical formula expects carrier frequency in MHz.
        f_c_mhz = self.f_c / 1e6
        L_dB = 40 * (1 - 4e-3 * h_mBS) * np.log10(d_3D) - 18 * np.log10(h_mBS) + 21 * np.log10(f_c_mhz) + 80
        return L_dB
    
    def get_snr(self, d_2D, h_mBS, p_tx_dbm, sigma_logf, shadowing_db=None):
        """
        Calculate linear SNR based on path loss, shadowing, and noise power.
        :param d_2D: 2D distance between mBS and user (m)
        :param h_mBS: Height of the mBS (m)
        :param p_tx_dbm: Transmit power (dBm)
        :param sigma_logf: Shadowing standard deviation (dB)
        :return: Linear SNR
        """
        L_dB = self.path_loss_db(d_2D, h_mBS)
        log_f = np.random.normal(0, sigma_logf) if shadowing_db is None else shadowing_db
        snr_db = p_tx_dbm - L_dB - self.sigma2_dbm - log_f
        snr_linear = 10 ** (snr_db / 10)
        return snr_linear
        
    