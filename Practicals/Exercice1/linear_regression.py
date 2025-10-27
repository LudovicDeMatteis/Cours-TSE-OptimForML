import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def f(x, a, b):
    return a*x + b

def generate_linear_data_samples(a, b, N=20, sigma=0.1, bias=0):
    sampling_points = np.linspace(0, 10, N)
    sampling_points += np.random.normal(0, 0.2, (N,))
    sampled_values = f(sampling_points, a, b)
    errors = np.random.normal(bias, sigma, (N,))
    sampled_values += errors
    return sampling_points, sampled_values

xis, yis = generate_linear_data_samples(a=2, b=2, sigma=2)
gd = f(xis, a=2, b=2)

## Estimate the solution via optimization
def J_val(a, b, xis, yis):
    return (1/2) * np.linalg.norm((a * xis + b) - yis)
J = lambda v : J_val(v[0], v[1], xis, yis)
# Define the function derivative
def dJ_val_dv(a, b, xis, yis):
    dJ_da = xis.T @ ((a * xis + b) - yis)
    dJ_db = np.ones_like(xis) @ ((a * xis + b) - yis)
    return np.array([dJ_da, dJ_db])

dJ_dv = lambda v: dJ_val_dv(v[0], v[1], xis, yis)

def solve_gradient_descent(a=1, b=1, max_iters=5000, tol=1e-3):
    ## Implement gradient descent the find the optimal solution
    num_iter = 0
    v = np.array([a, b])
    dJ_dguess = dJ_dv(v)
    alpha = 1e-3
    while np.linalg.norm(dJ_dguess) > tol and num_iter < max_iters:
        v = v - alpha * dJ_dguess
        num_iter += 1
        dJ_dguess =  dJ_dv(v)
    print(f"Gradient descent converged to {v} in {num_iter} iterations")
    return v

v = solve_gradient_descent()
guess = f(xis, v[0], v[1])

def ddJ_val_dv(a, b, xis, yis):
    ddJ_daa = xis.T @ xis
    ddJ_dbb = np.ones_like(xis).T @ np.ones_like(xis)
    ddJ_dab = xis.T @ np.ones_like(xis)
    return np.array([[ddJ_daa, ddJ_dab],
                     [ddJ_dab, ddJ_dbb]])
ddJ_dv = lambda v: ddJ_val_dv(v[0], v[1], xis, yis)

def solve_newton(a=1, b=1, max_iters=5000, tol=1e-3):
    ## Implement gradient descent the find the optimal solution
    num_iter = 0
    v = np.array([a, b])
    dJ_dguess = dJ_dv(v)
    ddJ_dguess = ddJ_dv(v)
    alpha = 1e0
    while np.linalg.norm(dJ_dguess) > tol and num_iter < max_iters:
        v = v - alpha * np.linalg.inv(ddJ_dguess) @ dJ_dguess
        num_iter += 1
        dJ_dguess = dJ_dv(v)
        ddJ_dguess = ddJ_dv(v)
    print(f"Newton's method converged to {v} in {num_iter} iterations")
    return v

v_newt = solve_newton()
guess_newt = f(xis, v[0], v[1])

plt.figure()
plt.scatter(xis, yis)
plt.plot(xis, gd, "r-.")
plt.plot(xis, guess, "b--")
plt.plot(xis, guess_newt, "g--")
plt.show()

