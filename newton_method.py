import numpy as np
from functions import (
    func_1,
    actual_gradient_func1,
    hessian_func1,
)


def newton_method(
    func,
    hessian_function,
    gradient_function,
    initial_x,
    initial_alpha,
    delta,
    maximum_interation,
    iteration,
):
    iteration = iteration + 1

    grad = gradient_function(initial_x)
    p_vector = -np.linalg.solve(hessian_function(), grad)
    
    new_x = initial_x + (initial_alpha * p_vector)

    difference = abs(func(initial_x) - func(new_x))
    if (difference < delta) or (iteration >= maximum_interation):
        return new_x

    else:
        return newton_method(
            func,
            hessian_function,
            gradient_function,
            initial_x=new_x,
            initial_alpha=initial_alpha,
            delta=delta,
            maximum_interation=maximum_interation,
            iteration=iteration,
        )


if __name__ == "__main__":
    optimum_x_value1 = newton_method(
        func_1,
        hessian_func1,
        actual_gradient_func1,
        initial_x=np.array([50, 50]),
        initial_alpha=1,
        delta=0.001,
        maximum_interation=100,
        iteration=1,
    )
