import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

class NeuralNet:
    def __init__(self, n_in, n_mid, n_out):
        self.yi, self.yj, self.yk = np.zeros(n_in), np.zeros(n_mid), np.zeros(n_out)
        self.weight_mid = np.random.random_sample((n_in,  n_mid))
        self.weight_out = np.random.random_sample((n_mid, n_out))
        self.d_mid = np.zeros((n_in,  n_mid))
        self.d_out = np.zeros((n_mid, n_out))
        self.teacher = np.eye((n_out))
        self.eta = 0.8
        self.alpha = 0.5
    
    def weight_update(self, v):
        middle_vec = v*self.yk*(1-self.yk)
        out_vec    = ((self.weight_out * middle_vec) @ np.ones(20))* self.yj*(1 - self.yj)
        d_out = np.array( np.matrix(self.yj).T * middle_vec)
        d_mid = np.array( np.matrix(self.yi).T * out_vec)
        self.d_mid = self.eta * d_mid + self.alpha * self.d_mid
        self.d_out = self.eta * d_out + self.alpha * self.d_out
        self.weight_mid += self.d_mid
        self.weight_out += self.d_out

    def learn(self, dataset):
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                t = self.teacher[i] - self.backforward(dataset[i][j])
                self.weight_update(t)

        
    def backforward(self, data):
        self.yi = data
        self.yj = sigmoid(self.yi @ self.weight_mid)
        self.yk = sigmoid(self.yj @ self.weight_out)
        return self.yk

    def evaluate(self, dataset):
        s = 0
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                t = self.teacher[i] - self.backforward(dataset[i][j])
                s += np.sum((t * t))/len(t)
        s /= dataset.shape[0]
        return s

    def predict(self, data):
        y = self.backforward(data)
        return np.argmax(y)
        
    def test(self, dataset):
        correct_ans_rate = []
        w, h = dataset.shape[0], dataset.shape[1]
        for i in range(w):
            count = 0
            for j in range(h):
                if(self.predict(dataset[i][j]) == i):
                    count += 1
            correct_ans_rate.append((count / h)*100)
        average = np.array(correct_ans_rate).sum() / w
        # correct_ans_rate.append(np.array(correct_ans_rate).sum() / w)
        return average