import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n_features, hidden_units=12):
    np.random.seed(42) 
    
    # ReLU hidden layer
    W1 = np.random.randn(n_features, hidden_units) * np.sqrt(2. / n_features)
    b1 = np.zeros((1, hidden_units))
    # Sigmoid output layer
    W2 = np.random.randn(hidden_units, 1) * 0.01
    b2 = np.zeros((1, 1))
    
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    # Layer 1 (Hidden Layer) using ReLU activation
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1) 
    
    # Layer 2 (Output Layer) sigmoid for binary probability output
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    return Z1, A1, A2

def compute_cost(y, A2):
    m = y.shape[0]
    epsilon = 1e-15  
    A2 = np.clip(A2, epsilon, 1 - epsilon)
    cost = (1 / m) * np.sum(-y * np.log(A2) - (1 - y) * np.log(1 - A2))
    return cost

def compute_gradients(X, y, Z1, A1, A2, W2):
    m = y.shape[0]
    
    # Output layer gradients (Backward Pass Step 1) 
    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden layer gradients using Chain Rule + ReLU Derivative
    relu_derivative = (Z1 > 0).astype(float)
    dZ1 = np.dot(dZ2, W2.T) * relu_derivative
    
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2

def predict_prob(X, W1, b1, W2, b2):
    Z1, A1, A2 = forward_propagation(X, W1, b1, W2, b2)
    return A2

def predict(X, W1, b1, W2, b2, threshold=0.5):
    probabilities = predict_prob(X, W1, b1, W2, b2)
    return (probabilities >= threshold).astype(int)

def gradient_descent(X, y, learning_rate=0.1, num_iterations=10000, hidden_units=12):
    loss = []
    m, n = X.shape
    W1, b1, W2, b2 = initialize_parameters(n, hidden_units)

    for i in range(num_iterations):
        # Forward pass 
        Z1, A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        cost = compute_cost(y, A2)
        loss.append(cost)

        # Backward pass 
        dW1, db1, dW2, db2 = compute_gradients(X, y, Z1, A1, A2, W2)
        
        # Update all parameters simultaneously
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return W1, b1, W2, b2, loss

if __name__ == "__main__":
    
    df = pd.read_csv('data.csv')
    X = df[['Age', 'EstimatedSalary']].values
    y = df['Purchased'].values.reshape(-1, 1)
    
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print("Training Neural Network (ReLU, 12 Hidden Units)...")
    W1, b1, W2, b2, history = gradient_descent(
        X_scaled, y, 
        learning_rate=0.1, 
        num_iterations=10000, 
        hidden_units=12
    )
    
    predictions = predict(X_scaled, W1, b1, W2, b2)
    accuracy = np.mean(predictions == y) * 100
    print(f"Final Neural Network Accuracy: {accuracy:.2f}%")

    plt.plot(range(len(history)), history, color='red')
    plt.title('Neural Network Cost History over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Binary Cross-Entropy)')
    plt.show()