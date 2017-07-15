# ============
# Set rewards
# ============

import scipy.signal

class Reward(object):
    """

    """
    def __init__(self, factor, gamma):
        self.factor = factor
        self.gamma = gamma

    # set step rewards to total episode reward
    def total(self, ep_batch, tot_reward):
        for step in ep_batch:
            step[2] = tot_reward * self.factor
        return ep_batch

    # set step rewards to discounted reward
    def discount(self, ep_batch):
        x = ep_batch[:, 2]

        discounted = scipy.signal.lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]
        discounted *= self.factor

        for i in range(len(discounted)):
            ep_batch[i, 2] = discounted[i]

        return ep_batch
