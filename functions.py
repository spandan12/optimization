import numpy as np


def func_1(x):
    return x[0] ** 2 + 10 * x[1] ** 2


def actual_gradient_func1(x):
    return np.array([2 * x[0], 20 * x[1]]).reshape(-1,1)


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
    ).reshape(-1,1)


def hessian_func1():
    return np.array([[2, 0], [0, 20]])

def problem_function_1(x):
    func_value = 0

    for i in range(0, 100):
        update_value = (i+1) * (x[i] ** 2)
        func_value = func_value + update_value

    return func_value



