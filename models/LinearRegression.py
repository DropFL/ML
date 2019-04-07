import numpy as np
import math

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = 0   # loss of final epoch

        # Training should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================

        num_batches = math.ceil(len(y) / batch_size)
        concat = np.concatenate((x, y[:, None]), axis=1)

        for e in range(epochs):
            final_loss = 0

            np.random.shuffle(concat)
            
            new_x = concat[:, :self.num_features]
            new_y = concat[:,  self.num_features]
            
            for b in range(num_batches):
                train_x = new_x[b*batch_size : (b+1)*batch_size]
                train_y = new_y[b*batch_size : (b+1)*batch_size][:, None]

                empir_y = np.dot(train_x, self.W)
                err_y = empir_y - train_y

                dW = np.transpose(train_x).dot(err_y) / batch_size * 2
                final_loss += (err_y ** 2).mean()

                self.W = optim.update(self.W, dW, lr)
            
            final_loss /= num_batches
        # ============================================================
        return final_loss

    def eval(self, x):
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        pred = np.dot(x, self.W)
        # ============================================================
        return pred