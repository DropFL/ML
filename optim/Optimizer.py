import numpy as np

"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!! 

[Description]
__init__ - Initialize necessary variables for optimizer class
input   : gamma, epsilon
return  : X

update   - Update weight for one minibatch
input   : w - current weight, grad - gradient for w, lr - learning rate
return  : updated weight 
"""

class SGD:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        pass
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        updated_weight = w - grad * lr
        # =============================================================
        return updated_weight

class Momentum:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.prev = None
        self.gamma = gamma
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        if self.prev is None:
            self.prev = lr * grad
        else:
            self.prev = self.prev * self.gamma + lr * grad
        updated_weight = w - self.prev
        # =============================================================
        return updated_weight


class RMSProp:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        self.gain = None
        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        if self.gain is None:
            self.gain = (1 - self.gamma) * (grad ** 2)
        else:
            self.gain = self.gamma * self.gain + (1 - self.gamma) * (grad ** 2)

        updated_weight = w - (lr / (self.gain + self.epsilon) ** 0.5) * grad
        # =============================================================
        return updated_weight