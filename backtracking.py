import numpy as np
from functions import func_1, actual_gradient_func1, get_a_b_c_coefficient


def backtracking(func, grad_func, p_vector, x_vector, initial_alpha, rho, c):
    grad = grad_func(x_vector)

    lhs = func(x_vector + initial_alpha * p_vector)

    rhs = func(x_vector) + c * initial_alpha * (p_vector.T @ grad)

    if lhs <= rhs:
        return initial_alpha
    else:
        updated_alpha = rho * initial_alpha
        return backtracking(func, grad_func, p_vector, x_vector, updated_alpha, rho, c)


def constraint_backtracking(func, grad_func, p_vector, x_vector, initial_alpha, rho, c):
    # grad = grad_func(x_vector)

    # lhs = func(x_vector + initial_alpha * p_vector)

    # rhs = func(x_vector) + c * initial_alpha * (p_vector.T @ grad)

    A, B, C = get_a_b_c_coefficient()

    constraint_value = np.array(B - A @ (x_vector + initial_alpha * p_vector))
    # breakpoint()
    if (not np.all(constraint_value > 0.0) ):
        updated_alpha = rho * initial_alpha
        return constraint_backtracking(func, grad_func, p_vector, x_vector, updated_alpha, rho, c)
        
    # else:
    return backtracking(func, grad_func, p_vector, x_vector, initial_alpha, rho, c)
        


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
