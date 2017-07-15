# =========
# Noise
# =========

import numpy as np

class Noise(object):
    """
    
    """
    def __init__(self, delta, sigma, ou_a, ou_mu):
        # noise parameters
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu

    def brownian_motion_log_returns(self):
        # This method returns a Wiener process
        # wiki: http://en.wikipedia.org/wiki/Wiener_process
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=None)

    def ornstein_uhlenbeck_level(self, prev_ou_level):
        # This method returns the rate levels of a mean-reverting ornstein uhlenbeck process
        drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
        randomness = self.brownian_motion_log_returns()
        return prev_ou_level + drift + randomness
