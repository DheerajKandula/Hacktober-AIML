# TASK - 2

# 1. Implement linear regression with gradient descent
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.b = 0
        self.cost_history = []
    
    def fit(self, X, y):
        n = len(X)
        
        for epoch in range(self.epochs):
            y_pred = self.m * X + self.b
            
            cost = (1/n) * np.sum((y - y_pred)**2)
            self.cost_history.append(cost)
            
            dm = -(2/n) * np.sum(X * (y - y_pred))
            db = -(2/n) * np.sum(y - y_pred)
            
            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
        
        print(f"\nFinal Parameters: slope = {self.m:.4f}, intercept = {self.b:.4f}")
    
    def predict(self, X):
        return self.m * X + self.b
    
    def plot_results(self, X, y):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.scatter(X, y, color='blue', alpha=0.6, label='Data points')
        ax1.plot(X, self.predict(X), color='red', linewidth=2, label='Best fit line')
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title('Linear Regression Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.cost_history, color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Cost (MSE)')
        ax2.set_title('Cost Function Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def LRExample():
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = 2.5 * X + 1 + np.random.randn(10) * 2
    
    model = LinearRegressionGD(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    model.plot_results(X, y)
    
    X_new = np.array([11, 12])
    print(f"\nPredictions for {X_new}: {model.predict(X_new)}")

# 2. Implement a Perceptron for binary classification

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for epoch in range(self.epochs):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._step_function(linear_output)
                
                if y[idx] != y_predicted:
                    update = self.learning_rate * (y[idx] - y_predicted)
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            self.errors.append(errors)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Errors = {errors}")
        
        print(f"\nTraining Complete! Final weights: {self.weights}, bias: {self.bias:.4f}")
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._step_function(linear_output)
    
    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def plot_results(self, X, y):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', marker='o', label='Class 0')
        ax1.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', marker='x', label='Class 1')
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        if self.weights[1] != 0:
            x_line = np.array([x_min, x_max])
            y_line = -(self.weights[0] * x_line + self.bias) / self.weights[1]
            ax1.plot(x_line, y_line, 'k-', linewidth=2, label='Decision Boundary')
        
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('Perceptron Classification')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(len(self.errors)), self.errors, color='green', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Errors')
        ax2.set_title('Training Errors Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def Perceptron_example():
    np.random.seed(42)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    y_and = np.array([0, 0, 0, 1])
    
    print("Training Perceptron on OR function:")
    perceptron_or = Perceptron(learning_rate=0.1, epochs=20)
    perceptron_or.fit(X, y_or)
    
    print("\nTesting OR Perceptron:")
    for x_i, target in zip(X, y_or):
        prediction = perceptron_or.predict(x_i.reshape(1, -1))
        print(f"Input: {x_i}, Target: {target}, Prediction: {prediction[0]}")
    
    print("\n" + "="*50)
    print("\nTraining Perceptron on AND function:")
    perceptron_and = Perceptron(learning_rate=0.1, epochs=20)
    perceptron_and.fit(X, y_and)
    
    print("\nTesting AND Perceptron:")
    for x_i, target in zip(X, y_and):
        prediction = perceptron_and.predict(x_i.reshape(1, -1))
        print(f"Input: {x_i}, Target: {target}, Prediction: {prediction[0]}")
    
    X_random = np.random.randn(100, 2)
    y_random = (X_random[:, 0] + X_random[:, 1] > 0).astype(int)
    
    perceptron_random = Perceptron(learning_rate=0.1, epochs=100)
    perceptron_random.fit(X_random, y_random)
    perceptron_random.plot_results(X_random, y_random)

# 3. 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from perceptron import Perceptron

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin_model = LinearRegression(learning_rate=0.1, epochs=1000)
    start_train = time.time()
    lin_model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_pred_lin = lin_model.predict(X_test)
    y_pred_lin = np.where(y_pred_lin >= 0.5, 1, 0)
    end_pred = time.time()

    acc_lin = accuracy(y_test, y_pred_lin)
    time_to_converge_lin = end_train - start_train
    time_per_pred_lin = (end_pred - start_pred) / len(y_test)

    perc_model = Perceptron(learning_rate=0.01, epochs=1000)
    start_train = time.time()
    perc_model.fit(X_train, y_train)
    end_train = time.time()

    start_pred = time.time()
    y_pred_perc = perc_model.predict(X_test)
    end_pred = time.time()

    acc_perc = accuracy(y_test, y_pred_perc)
    time_to_converge_perc = end_train - start_train
    time_per_pred_perc = (end_pred - start_pred) / len(y_test)

    results = {
        "LinearRegression": {
            "accuracy": float(acc_lin),
            "time_to_convergence": float(time_to_converge_lin),
            "time_per_prediction": float(time_per_pred_lin)
        },
        "Perceptron": {
            "accuracy": float(acc_perc),
            "time_to_convergence": float(time_to_converge_perc),
            "time_per_prediction": float(time_per_pred_perc)
        }
    }
    
    print("\nModel Comparison Results:")
    print("="*50)
    print(f"Linear Regression - Accuracy: {acc_lin*100:.2f}%")
    print(f"Linear Regression - Time to Convergence: {time_to_converge_lin:.6f}s")
    print(f"Linear Regression - Time per Prediction: {time_per_pred_lin:.6f}s")
    print("-"*50)
    print(f"Perceptron - Accuracy: {acc_perc*100:.2f}%")
    print(f"Perceptron - Time to Convergence: {time_to_converge_perc:.6f}s")
    print(f"Perceptron - Time per Prediction: {time_per_pred_perc:.6f}s")
    print("="*50)
    
    return results

data = pd.read_csv('binary_classification.csv')
X = data[['x1', 'x2']].values
y = data['label'].values

results = evaluate_models(X, y)

with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to model_comparison_results.json")


# 4. Log the following metrics: Accuracy, Time to convergence, Time per prediction
import time

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(X):
                prediction = 1 if np.dot(x_i, self.weights) + self.bias >= 0 else 0
                
                if y[idx] != prediction:
                    update = self.learning_rate * (y[idx] - prediction)
                    self.weights += update * x_i
                    self.bias += update
        
        convergence_time = time.time() - start_time
        return convergence_time
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias >= 0, 1, 0)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = Perceptron()
convergence_time = model.fit(X, y)

start_pred = time.perf_counter()
predictions = model.predict(X)
pred_time = (time.perf_counter() - start_pred) * 1000

acc = model.accuracy(X, y)

print("METRICS")
print("="*40)
print(f"Accuracy: {acc*100:.2f}%")
print(f"Time to Convergence: {convergence_time:.6f} seconds")
print(f"Time per Prediction: {pred_time:.6f} ms")
print("="*40)