import pandas as pd
import numpy as np

df = pd.read_csv('Social_Network_Ads.csv')
df = df.drop(['User ID', 'Gender'], axis=1)

X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values.reshape(-1, 1)

X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n_features):
    w = np.zeros((n_features, 1))
    b = 0
    return w, b

def predict_prob(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def predict(X, w, b, threshold=0.5):
    probabilities = predict_prob(X, w, b)
    return (probabilities >= threshold).astype(int)

def compute_cost(y, A):
    m = y.shape[0]
    epsilon = 1e-15  # To avoid log(0)
    A = np.clip(A, epsilon, 1 - epsilon)  # Clip A to avoid log(0)
    cost = (1 / m) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
    return cost

def compute_gradient(X, y, A):
    m = y.shape[0]
    dw = (1 / m) * np.dot(X.T, (A - y))
    db = (1 / m) * np.sum(A - y)
    return dw, db

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    loss = []
    m, n = X.shape
    w, b = initialize_parameters(n)

    for i in range(num_iterations):
        A = predict_prob(X, w, b)
        cost = compute_cost(y, A)
        loss.append(cost)

        dw, db = compute_gradient(X, y, A)
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, loss
