import numpy as np
import math

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================

        num_batches = math.ceil(len(y) / batch_size)
        concat = np.concatenate((x, y[:, None]), axis=1)
        final_loss = 0
        
        for e in range(epochs):

            np.random.shuffle(concat)
            
            new_x = concat[:, :self.num_features]
            new_y = concat[:,  self.num_features]
            
            for b in range(num_batches):
                train_x = new_x[b*batch_size : (b+1)*batch_size]
                train_y = new_y[b*batch_size : (b+1)*batch_size][:, None]

                empir_y = self._sigmoid(train_x)
                err_y = empir_y - train_y

                dW = np.transpose(train_x).dot(err_y) / batch_size * 2
                
                if e == epochs - 1:
                    # final_loss -= (train_y * empir_y - np.log(1 + np.exp(empir_y))).mean()
                    final_loss -= (train_y * np.log(empir_y)).sum()

                self.W = optim.update(self.W, dW, lr)
        final_loss /= num_batches

        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================

        pred = self._sigmoid(x)
        
        for p in np.nditer(pred, op_flags=['readwrite']):
            p[...] = 1 if p > threshold else 0

        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================

        sigmoid = 1 / (np.exp(-np.dot(x, self.W)) + 1)

        # ============================================================
        return sigmoid