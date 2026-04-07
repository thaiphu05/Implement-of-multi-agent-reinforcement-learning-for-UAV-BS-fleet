import numpy as np

class Channel_Model_UAV:
    def __init__(self, f_c=2e9, alpha=2.5, sigma2_dbm=-110, k_factor=10.0):
        """
        :param f_c: Carrier frequency (Hz)
        :param alpha: path-loss exponent
        :param sigma2_dbm: Noise power (dBm)
        :param k_factor: Rician K-factor (linear), LOS-dominant when K > 1
        """
        self.f_c = f_c
        self.alpha = alpha
        self.k_factor = float(k_factor)
        self.c = 3e8
        self.lamda = self.c / self.f_c
        self.d_ref = 1.0

        # Friis reference gain at d_ref.
        theta_db = -20 * np.log10(4 * np.pi * self.d_ref / self.lamda)
        self.theta_linear = 10 ** (theta_db / 10)
        self.sigma2 = 10 ** ((sigma2_dbm - 30) / 10)

    def _sample_rician_power(self):
        """Sample |h|^2 for a unit-average-power Rician channel."""
        k = max(self.k_factor, 0.0)
        los_amp = np.sqrt(k / (k + 1.0))
        nlos_scale = np.sqrt(1.0 / (2.0 * (k + 1.0)))
        h = (los_amp + np.random.normal(0.0, nlos_scale)) + 1j * np.random.normal(0.0, nlos_scale)
        return float(np.abs(h) ** 2)

    def sample_fading_power(self, size=None):
        """Sample Rician fading power |h|^2 with optional output shape."""
        if size is None:
            return self._sample_rician_power()
        samples = np.empty(size, dtype=np.float32)
        for idx in np.ndindex(samples.shape):
            samples[idx] = self._sample_rician_power()
        return samples

    def get_path_loss_gain(self, d_2D, h_UAV, fading_power=None):
        """
        Calculate the path loss gain based on the distance between the UAV and the user.
        :param d_2D: Distance between UAV and user (m)
        :param h_UAV: UAV height (m)
        :return: Path loss gain
        """
        d_3D = np.sqrt(d_2D**2 + h_UAV**2)

        if fading_power is None:
            
            fading_power = self._sample_rician_power()

        path_loss_gain = fading_power * self.theta_linear * ((d_3D / self.d_ref) ** (-self.alpha))

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
        p_tx_w = 10 ** ((p_tx_dbm - 30) / 10)
        snr_linear = p_tx_w * path_loss_gain / self.sigma2
        return snr_linear


class Channel_Model_mBS:
    def __init__(self, f_c=2e9, sigma2_dbm=-90, d_ref=1):
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
        d_3D = np.sqrt(d_2D**2 + h_mBS**2) / 1000.0
        # Avoid log10(0) when user is at the same horizontal position as mBS.
        d_3D = np.maximum(d_3D, 1e-6)
        # The empirical formula expects carrier frequency in MHz.
        f_c_mhz = self.f_c / 1e6
        L_dB = 40 * (1 - 4e-3 * h_mBS) * np.log10(d_3D) - 18 * np.log10(h_mBS) + 21 * np.log10(f_c_mhz) + 80
        return L_dB
    
    def get_snr(self, d_2D, h_mBS, p_tx_dbm, sigma_logf, shadowing_db=None):
        """
        Calculate linear SNR based on the paper's mBS link model.
        :param d_2D: 2D distance between mBS and user (m)
        :param h_mBS: Height of the mBS (m)
        :param p_tx_dbm: Transmit power (dBm)
        :param sigma_logf: Shadowing standard deviation (dB)
        :return: Linear SNR
        """
        L_dB = self.path_loss_db(d_2D, h_mBS)
        log_f = np.random.normal(0, sigma_logf) if shadowing_db is None else shadowing_db
        snr_db = p_tx_dbm - L_dB - log_f - 8 - self.sigma2_dbm
        snr_linear = 10 ** (snr_db / 10)
        return snr_linear
        
    