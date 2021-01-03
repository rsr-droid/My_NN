from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def plot(xfeature_name, yfeature_name, xfeature, yfeature, ori_label, re_label):
    # setting color
    ori_color = {0: 'red', 1: 'blue', 2: 'yellow'} # label, 0:'setosa' 1:'versicolor' 2:'virginica'
    relbl_color = {0: 'blue', 1: 'red'} # relabel, 0:'not setosa' 1:'setosa'
    # plot oringinal label figure
    plt.figure(figsize=(15,8))
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(wspace = 0.5)
    plt.title("Original Label")
    plt.xlabel(xfeature_name)
    plt.ylabel(yfeature_name)
    for i, j, color in zip(xfeature, yfeature, ori_label):
        plt.scatter(i, j, c=ori_color[color], marker='o', s=50, edgecolor='k',cmap=plt.cm.Spectral)
    # plot relabel figure
    plt.subplot(1, 2, 2)
    plt.title("Relabel")
    plt.xlabel(xfeature_name)
    plt.ylabel(yfeature_name)
    for i, j, color in zip(xfeature, yfeature, re_label):
        plt.scatter(i, j, c=relbl_color[color], marker='o', s=50, edgecolor='k',cmap=plt.cm.Spectral)


class my_NN(object):
    def __init__(self):
        # initialize input, output, hidden units
        self.input_units = 4 # feature numbers
        self.output_units = 1 # class number
        self.hidden_units = 6 # single layer
        
        # initialize random values for weight matrix
        np.random.seed(1)
        
        # weight1: input -> hidden layer
        self.w1 = np.random.randn(self.input_units, self.hidden_units) #input x hidden matrix = 4x6
        
        # weight2: hidden layer -> output layer
        self.w2 = np.random.randn(self.hidden_units, self.output_units) #hidden x output matrux = 6x1
        
    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1.T, X.T)
        self.a2 = self._sigmoid(self.z2)
        self.z3 = np.dot(self.w2.T, self.a2)
        self.a3 = self._sigmoid(self.z3)
        return self.a3
        
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def _loss(self, predict, y):
        m = y.shape[0]
        loss = (-1/m) * np.sum((np.log(predict) * y) + np.log(1 - predict) * (1 - y))
        return loss

    # Compute the Partial Derivative of J(Θ).
    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X) 
        m = X.shape[0]
        delta3 = predict - y # 1x150
        dz3 = np.multiply(delta3, self._sigmoid_prime(self.z3))
        self.dw2 = (1/m)*np.dot(dz3, self.a2.T).reshape(self.w2.shape)

        delta2 = delta2 = delta3*clr.w2*self._sigmoid_prime(self.z2)
        self.dw1 = (1/m)*np.dot(delta2, X).reshape(self.w1.shape)
    
    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))
    
    # Update the weighting matrix W¹ and W² with the result of the Partial Derivative of J(Θ).
    def _update(self, learning_rate=1.9):
        self.w1 = self.w1 - learning_rate*self.dw1
        self.w2 = self.w2 - learning_rate*self.dw2
        
    def train(self, X, y, iteration=60):
        for i in range(iteration):
            # compute forward prop (y_hat)
            y_hat = self._forward_propagation(X)
            # return loss
            loss = self._loss(y_hat, y)
            # compute back prop
            self._backward_propagation(X, y)
            # update weights
            self._update()
            if i % 10 == 0:
                print("loss: ", loss)
        
    def predict(self, X):
        # compute forward prop again with newly updated weights (y_hat)
        # set y_hat as either 1 or 0 depending on >0.5 or <0.5
        y_hat = self._forward_propagation(X)
        y_hat = [1 if i[0] >= 0.5 else 0 for i in y_hat.T]
        return np.array(y_hat)
        
    def score(self, predict, y):
        # compute accuracy of prediction model
        cnt = np.sum(predict==y)
        return (cnt/len(y))*100
