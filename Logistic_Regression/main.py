import numpy as np

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

def gradient_descent(X, y, learning_rate=0.01, num_iterations=5000):
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

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('data.csv')
    X = df[['Age', 'EstimatedSalary']].values
    y = df['Purchased'].values.reshape(-1, 1)
    
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print("Training model...")
    w_trained, b_trained, history = gradient_descent(X_scaled, y, learning_rate=0.1, num_iterations=5000)
    
    predictions = predict(X_scaled, w_trained, b_trained)
    accuracy = np.mean(predictions == y) * 100
    print(f"Final Model Accuracy: {accuracy:.2f}%")
 
    plt.plot(range(len(history)), history, color='red')
    plt.title('Cost History over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
