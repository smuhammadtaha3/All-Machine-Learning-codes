import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import pickle
from utils import *
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROJECT_ROOT = os.path.dirname(BASE_DIR)

BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
METADATA_DIR = os.path.join(BACKEND_DIR, "metadata")
STATIC_DIR = os.path.join(BACKEND_DIR, "static")

# CREATE FOLDERS IF THEY DON'T EXIST
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ==========================================
# PART 1: LOGISTIC REGRESSION
# ==========================================

# Load dataset 1
data1_path = os.path.join(DATA_DIR, "ex2data1.txt")
X_train, y_train = load_data(data1_path) # [cite_start]X_train, y_train = load_data("ex2data1.txt") [cite: 3]

def sigmoid(z):
    
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    loss_sum = 0.0
    for i in range(m):
        z_wb = np.dot(X[i], w) + b    
        f_wb = sigmoid(z_wb)
        loss = (-y[i] * np.log(f_wb)) - ((1-y[i]) * np.log(1 - f_wb))
        loss_sum += loss
    return (1/m) * loss_sum

def compute_gradient(X, y, w, b, *argv): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        err = f_wb - y[i]
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i][j]
    return dj_db/m, dj_dw/m

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)   
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db                     
        if i % 1000 == 0:
            cost = cost_function(X, y, w, b, lambda_)
            print(f"Iteration {i:4}: Cost {float(cost):8.2f}")
    return w, b

# ==========================================
# PART 2: REGULARIZED LOGISTIC REGRESSION (Actual Training)
# ==========================================

# [cite_start]Load dataset 2 for advanced training [cite: 3]
data2_path = os.path.join(DATA_DIR, "ex2data2.txt")
X_train2, y_train2 = load_data(data2_path)
X_mapped = map_feature(X_train2[:, 0], X_train2[:, 1])

def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape    
    cost_without_reg = compute_cost(X, y, w, b)     
    reg_cost = np.sum(np.square(w))
    return cost_without_reg + (lambda_ / (2 * m)) * reg_cost

def compute_gradient_reg(X, y, w, b, lambda_=1): 
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    for j in range(n):
        dj_dw[j] += (lambda_/m) * w[j]
    return dj_db, dj_dw

# Training start
print("Training started...")
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 1.
lambda_ = 0.01    
iterations = 10000
alpha = 0.01

w, b, = gradient_descent(X_mapped, y_train2, initial_w, initial_b, 
                        compute_cost_reg, compute_gradient_reg, alpha, iterations, lambda_)

# ==========================================
# PART 3: SAVING OUTPUTS FOR WEBSITE (NEW CHANGES)
# ==========================================

def predict(X, w, b): 
    m = X.shape[0]   
    p = np.zeros(m)   
    for i in range(m):   
        z_wb = np.dot(X[i], w) + b        
        p[i] = sigmoid(z_wb) >= 0.5
    return p

# 1. Calculate and Save Accuracy
p_final = predict(X_mapped, w, b)
train_accuracy = np.mean(p_final == y_train2) * 100
print(f'Train Accuracy: {train_accuracy:.2f}%')

# Backend folder mein accuracy save karna
try:
    accuracy_path = os.path.join(METADATA_DIR, "accuracy.txt")

    with open(accuracy_path, "w") as f:
        f.write(f"{train_accuracy:.2f}")
    print("Accuracy file saved successfully.")
except Exception as e:
    print(f"Error saving accuracy file: {e}")

# 2. Save Decision Boundary Graph
plt.figure(figsize=(10, 8))
plot_decision_boundary(w, b, X_mapped, y_train2)
plt.ylabel('Microchip Test 2') 
plt.xlabel('Microchip Test 1') 
plt.title(f"Decision Boundary - Accuracy: {train_accuracy:.2f}%")
plt.legend(loc="upper right")

# Graph ko static folder mein save karna
graph_path = os.path.join(STATIC_DIR, "boundary_plot.png")
try:
    plt.savefig(graph_path)
    print(f"Graph saved successfully at: {graph_path}")
except Exception as e:
    print(f"Error saving graph: {e}. Make sure '../backend/static/' folder exists.")

# [cite_start]3. Save Model Pickle [cite: 1]
model_data = {"weights": w, "bias": b}
try:
    model_path = os.path.join(MODELS_DIR, "model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print("Model saved as model.pkl")
except Exception as e:
    print(f"Error saving model: {e}")

plt.show()