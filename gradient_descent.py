import numpy as np
from backtracking import backtracking
from functions import func_1, actual_gradient_func1, func_2, actual_gradient_func2


def gradient_descent(
    func,
    gradient_function,
    initial_x,
    initial_alpha,
    delta,
    maximum_interation,
    iteration,
):
    iteration = iteration + 1

    grad = gradient_function(initial_x)
    p_vector = -grad
    alpha = backtracking(
        func,
        gradient_function,
        p_vector=p_vector,
        x_vector=initial_x,
        initial_alpha=initial_alpha,
        rho=0.9,
        c=0.4,
    )

    new_x = initial_x + alpha * p_vector

    difference = np.linalg.norm(func(initial_x) - func(new_x))
    if (difference < delta) or (iteration >= maximum_interation):
        return new_x

    else:
        return gradient_descent(
            func,
            gradient_function,
            initial_x=new_x,
            initial_alpha=initial_alpha,
            delta=delta,
            maximum_interation=maximum_interation,
            iteration=iteration,
        )

# test functions
if __name__ == "__main__":
    optimum_x_value1 = gradient_descent(
        func_1,
        actual_gradient_func1,
        initial_x=np.array([50, 50]).reshape(-1,1),
        initial_alpha=0.01,
        delta=0.00001,
        maximum_interation=1000,
        iteration=1,
    )

    optimum_x_value2 = gradient_descent(
        func_2,
        actual_gradient_func2,
        initial_x=np.array([2, 1]).reshape(-1,1),
        initial_alpha=0.01,
        delta=0.001,
        maximum_interation=100,
        iteration=1,
    )
    print(optimum_x_value1, optimum_x_value2)
