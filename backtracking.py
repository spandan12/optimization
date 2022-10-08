import numpy as np
from functions import func_1, actual_gradient_func1


def backtracking(func, grad_func, p_vector, x_vector, initial_alpha, rho, c):
    grad = grad_func(x_vector)

    lhs = func(x_vector + initial_alpha * p_vector)

    rhs = func(x_vector) + c * initial_alpha * (p_vector.T @ grad)

    if lhs <= rhs:
        return initial_alpha
    else:
        updated_alpha = rho * initial_alpha
        return backtracking(func, grad_func, p_vector, x_vector, updated_alpha, rho, c)

# test function
if __name__ == "__main__":
    updated_alpha = backtracking(
        func_1,
        actual_gradient_func1,
        p_vector=np.array([1, 0]),
        x_vector=np.array([5, 5]),
        initial_alpha=20,
        rho=0.9,
        c=0.4,
    )
