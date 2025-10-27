import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def f(x, a, b, c):
    return a * x**3 + b * x + c

a_gt, b_gt, c_gt = 1, -2, 3
def generate_data(num_points=50, noise_std=5.0):
    x = np.linspace(-5, 5, num_points)
    x += np.random.normal(0, 0.4, size=num_points)
    x = np.sort(x)
    y = f(x, a_gt, b_gt, c_gt) + np.random.normal(0, noise_std, size=num_points)
    return x, y

xis, yis = generate_data()
gt = f(xis, a_gt, b_gt, c_gt)

def J_val(a, b, c, x, y):
    y_pred = f(x, a, b, c)
    return np.sum((y_pred - y) ** 2)
J = lambda v: J_val(v[0], v[1], v[2], xis, yis)
def J_grad(a, b, c, x, y):
    y_pred = f(x, a, b, c)
    error = y_pred - y
    dJ_da = 2 * np.sum(error * x**3)
    dJ_db = 2 * np.sum(error * x)
    dJ_dc = 2 * np.sum(error)
    return np.array([dJ_da, dJ_db, dJ_dc])
dJ = lambda v: J_grad(v[0], v[1], v[2], xis, yis)

def solve_gradient_descent(a0, b0, c0, learning_rate=1e-11, max_iters=50000, tol=1e-6):
    params = np.array([a0, b0, c0], dtype=float)
    for i in range(max_iters):
        grad = dJ(params)
        params_new = params - learning_rate * grad
        if np.linalg.norm(grad) < tol:
            break
        params = params_new
    print(f'Gradient Descent converged in {i+1} iterations.')
    return params

v_gd = solve_gradient_descent(-1.0, 1.0, 1.0)
guess_gd = f(xis, v_gd[0], v_gd[1], v_gd[2])

def J_hessian(a, b, c, x, y):
    y_pred = f(x, a, b, c)
    error = y_pred - y
    N = len(x)
    H = np.zeros((3, 3))
    H[0, 0] = 2 * np.sum(x**6)
    H[0, 1] = H[1, 0] = 2 * np.sum(x**4)
    H[0, 2] = H[2, 0] = 2 * np.sum(x**3)
    H[1, 1] = 2 * np.sum(x**2)
    H[1, 2] = H[2, 1] = 2 * np.sum(x)
    H[2, 2] = 2 * N
    return H
ddJ = lambda v: J_hessian(v[0], v[1], v[2], xis, yis)

def solve_newton(a0, b0, c0, max_iters=100, tol=1e-6):
    params = np.array([a0, b0, c0], dtype=float)
    for i in range(max_iters):
        grad = dJ(params)
        hess = ddJ(params)
        try:
            hess_inv = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping.")
            break
        params_new = params - hess_inv @ grad
        if np.linalg.norm(grad) < tol:
            break
        params = params_new
    print(f'Newton\'s Method converged in {i+1} iterations.')
    return params

v_newton = solve_newton(-1.0, 1.0, 1.0)
guess_newton = f(xis, v_newton[0], v_newton[1], v_newton[2])

plt.figure()
plt.scatter(xis, yis, label='Noisy Data', color='blue', s=10)
plt.plot(xis, gt, label='Ground Truth', color='green')
plt.plot(xis, guess_gd, label='Gradient Descent Fit', color='red')
plt.plot(xis, guess_newton, label='Newton Method Fit', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()

