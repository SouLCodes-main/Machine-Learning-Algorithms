import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_parameters(n_features, hidden_units=8):
    # Xavier-style initialization helps the hidden layer train more stably
    W1 = np.random.randn(n_features, hidden_units) * np.sqrt(2.0 / n_features)
    b1 = np.zeros((1, hidden_units))
    W2 = np.random.randn(hidden_units, 1) * np.sqrt(2.0 / hidden_units)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2


def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return A1, A2


def predict_prob(X, W1, b1, W2, b2):
    _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return A2


def predict(X, W1, b1, W2, b2, threshold=0.5):
    probabilities = predict_prob(X, W1, b1, W2, b2)
    return (probabilities >= threshold).astype(int)


def compute_cost(y, A2):
    m = y.shape[0]
    epsilon = 1e-15
    A2 = np.clip(A2, epsilon, 1 - epsilon)
    cost = (1 / m) * np.sum(-y * np.log(A2) - (1 - y) * np.log(1 - A2))
    return cost


def compute_gradients(X, y, A1, A2, W2):
    m = y.shape[0]

    dZ2 = A2 - y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def gradient_descent(X, y, learning_rate=0.05, num_iterations=10000, hidden_units=8):
    loss = []
    _, n = X.shape
    W1, b1, W2, b2 = initialize_parameters(n, hidden_units)

    for _ in range(num_iterations):
        A1, A2 = forward_propagation(X, W1, b1, W2, b2)
        loss.append(compute_cost(y, A2))

        dW1, db1, dW2, db2 = compute_gradients(X, y, A1, A2, W2)

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    return W1, b1, W2, b2, loss


if __name__ == "__main__":
    df = pd.read_csv('data.csv')
    X = df[['Age', 'EstimatedSalary']].values.astype(float)
    y = df['Purchased'].values.reshape(-1, 1).astype(float)

    rng = np.random.default_rng(42)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    split_index = int(0.8 * len(df))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print("Training Neural Network...")
    W1, b1, W2, b2, history = gradient_descent(
        X_train_scaled,
        y_train,
        learning_rate=0.05,
        num_iterations=10000,
        hidden_units=8,
    )

    train_predictions = predict(X_train_scaled, W1, b1, W2, b2)
    train_accuracy = np.mean(train_predictions == y_train) * 100

    test_predictions = predict(X_test_scaled, W1, b1, W2, b2)
    test_accuracy = np.mean(test_predictions == y_test) * 100

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    plt.plot(range(len(history)), history, color='red')
    plt.title('Neural Network Cost History over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('training_loss.png')
    plt.close()