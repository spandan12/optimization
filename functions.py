import numpy as np

from function_helper import read_coefficient_a, read_coefficient_b, read_coefficient_c

def func_1(x):
    return x[0] ** 2 + 10 * x[1] ** 2


def actual_gradient_func1(x):
    return np.array([2 * x[0], 20 * x[1]]).reshape(-1, 1)


def func_2(x):
    return (
        np.exp(x[0] + 3 * x[1] - 0.1)
        + np.exp(x[0] - 3 * x[1] - 0.1)
        + np.exp(-x[0] - 0.1)
    )


def actual_gradient_func2(x):
    return np.array(
        [
            np.exp(x[0] + 3 * x[1] - 0.1)
            + np.exp(x[0] - 3 * x[1] - 0.1)
            - np.exp(-x[0] - 0.1),
            3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
        ]
    ).reshape(-1, 1)


def hessian_func1(x):
    return np.array([[2, 0], [0, 20]])


# Project problem function 1
def problem_function_1(x):
    func_value = 0

    for i in range(0, 100):
        update_value = (i + 1) * (x[i] ** 2)
        func_value = func_value + update_value

    return func_value

# Project gradient function 1
def gradient_function_1(x):

    grad_func_value = np.array([(2 * (i + 1) * x[i]) for i in range(0, 100)]).reshape(
        -1, 1
    )

    return grad_func_value

def get_a_b_c_coefficient():
    return read_coefficient_a(), read_coefficient_b(), read_coefficient_c()

# Project problem function 2
def problem_function_2(x):

    A, B, C = get_a_b_c_coefficient()
    
    func_value = C.T @ x - np.sum(np.log(B - A @ x))
    

    return func_value

# Project gradient function 2
def gradient_function_2(x):

    A, B, C = get_a_b_c_coefficient()
    grad_func_value = C + A.T @ (np.reciprocal(B - A @ x))
    # breakpoint()
    return grad_func_value


# Project problem function 3
def problem_function_3(x):

    func_value = (100 * (x[1] - x[0] ** 2) ** 2) + (1 - x[0]) ** 2

    return func_value


# Project gradient function 3
def gradient_function_3(x):

    grad_func_value = np.array(
        [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
    ).reshape(-1, 1)

    return grad_func_value
