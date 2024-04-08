import numpy as np

class LinearRegression:
    def __init__(self):
        self.dim = 18 * 9 + 1
        self.w = np.zeros([self.dim, 1])
    
    def normalize(self, data):
        mean = np.mean(data, axis = 0)
        std = np.std(data, axis = 0)
        for i in range(len(data)):
            for j in range(len(data[0])):
                if std[j] != 0:
                    data[i][j] = (data[i][j] - mean[j]) / std[j]
        return data

    def fit(self, X, y, learning_rate, iterations, normalize = True):
        if normalize:
            X = self.normalize(X)
        x = np.concatenate((np.ones([12 * 471, 1]), X), axis = 1).astype(float)
        adagrad = np.zeros([self.dim, 1])
        eps = 0.0000000001
        for t in range(iterations):
            loss = np.sqrt(np.sum(np.power(np.dot(x, self.w) - y, 2))/471/12)#rmse
            if(t%100==0):
                print(str(t) + ":" + str(loss))
            gradient = 2 * np.dot(x.transpose(), np.dot(x, self.w) - y) #dim*1
            adagrad += gradient ** 2
            self.w = self.w - learning_rate * gradient / np.sqrt(adagrad + eps)
        #np.save('weight.npy', self.w)

    def predict(self, X, normalize = True):
        #w = np.load('weight.npy')
        # 預測
        if normalize:
            X = self.normalize(X)
        ans_y = np.dot(X, self.w)
        return ans_y