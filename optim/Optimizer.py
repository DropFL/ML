import numpy as np

class SGD:
    def __init__(self, gamma, epsilon):
        pass

    def update(self, w, grad, lr):
        return w - grad * lr

class Momentum:
    def __init__(self, gamma, epsilon):
        self.prev = None
        self.gamma = gamma

    def update(self, w, grad, lr):
        if self.prev is None:
            self.prev = lr * grad
        else:
            self.prev = self.prev * self.gamma + lr * grad
        return w - self.prev


class RMSProp:
    def __init__(self, gamma, epsilon):
        self.gamma = gamma
        self.epsilon = epsilon
        self.gain = None

    def update(self, w, grad, lr):
        updated_weight = None
        if self.gain is None:
            self.gain = (1 - self.gamma) * (grad ** 2)
        else:
            self.gain = self.gamma * self.gain + (1 - self.gamma) * (grad ** 2)

        updated_weight = w - (lr / (self.gain + self.epsilon) ** 0.5) * grad
        return updated_weight