import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Create synthetic data for linear classification using two Gaussian distributions
def generate_data(num_points_per_class=50, mean1=[2, 2], mean2=[5, 3], cov=[[0.7, 0.5], [0.5, 1]]):
    class1 = np.random.multivariate_normal(mean1, cov, num_points_per_class)
    class2 = np.random.multivariate_normal(mean2, cov, num_points_per_class)
    X = np.vstack((class1, class2))
    labels = np.array([0]*num_points_per_class + [1]*num_points_per_class)
    return X, labels

X, labels = generate_data()
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# Add bias term
X_aug = np.hstack((np.ones((X.shape[0], 1)), X))  # shape (N, 3)

# Visualize the generated data
plt.figure()
plt.title("Generated Data")
plt.scatter(*X.T, c=labels, cmap='bwr', edgecolor='k')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# Define the logistic regression model
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(X, weights):
    return sigmoid(np.dot(X, weights))

# Define the loss function (binary cross-entropy)
def J_val(weights, X, labels):
    m = len(labels)
    y_pred = model(X, weights)
    loss = - (1/m) * np.sum(labels * np.log(y_pred + 1e-15) + (1 - labels) * np.log(1 - y_pred + 1e-15))
    return loss
J = lambda w: J_val(w, X_aug, labels)

def J_grad(weights, X, labels):
    m = len(labels)
    y_pred = model(X, weights)
    gradient = (1/m) * np.dot(X.T, (y_pred - labels))
    return gradient
dJ  = lambda w: J_grad(w, X_aug, labels)

def J_hessian(weights, X, labels):
    m = len(labels)
    y_pred = model(X, weights)
    diag = y_pred * (1 - y_pred)
    H = (1/m) * np.dot(X.T, diag[:, None] * X)
    return H
ddJ = lambda w: J_hessian(w, X_aug, labels)

def solve_newton(weights0, max_iters=100, tol=1e-6):
    weights = weights0.copy()
    for i in range(max_iters):
        grad = dJ(weights)
        hess = ddJ(weights)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping optimization.")
            break
        weights_new = weights - delta
        if np.linalg.norm(grad) < tol:
            break
        weights = weights_new
    print(f'Newton\'s Method converged in {i+1} iterations.')
    return weights

w_newton = solve_newton(np.zeros(X_aug.shape[1]))
labels_pred_newton = model(X_aug, w_newton)

# Sort the values depending on the correct or incorrect prediction
X_00 = X[(labels == 0) & (labels_pred_newton < 0.5)]
X_11 = X[(labels == 1) & (labels_pred_newton >= 0.5)]
X_01 = X[(labels == 0) & (labels_pred_newton >= 0.5)]
X_10 = X[(labels == 1) & (labels_pred_newton < 0.5)]

# Plot the values with the decision boundary
x_vals = np.linspace(x_min, x_max, 100)
y_vals = -(w_newton[0] + w_newton[1] * x_vals) / w_newton[2]

plt.figure()
plt.title("Logistic Regression Decision Boundary")
plt.scatter(X_00[:, 0], X_00[:, 1], c='blue', marker='o', label='Class 0 Correct')
plt.scatter(X_11[:, 0], X_11[:, 1], c='red', marker='o', label='Class 1 Correct')
plt.scatter(X_01[:, 0], X_01[:, 1], c='blue', marker='x', label='Class 0 Incorrect')
plt.scatter(X_10[:, 0], X_10[:, 1], c='red', marker='x', label='Class 1 Incorrect')
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

