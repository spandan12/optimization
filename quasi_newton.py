import numpy as np
from functions import func_1, actual_gradient_func1
from backtracking import backtracking, constraint_backtracking


def quasi_newton(
    func,
    gradient_function,
    F_value,
    initial_x,
    initial_alpha,
    delta,
    maximum_interation,
    iteration,
    convergence,
    constraint
):
    # print(iteration)
    iteration = iteration + 1
    
    grad = gradient_function(initial_x)
    p_vector = - F_value @ grad
    
    if constraint == True:
        # print("i",initial_alpha)
        alpha = constraint_backtracking(
        func,
        gradient_function,
        p_vector=p_vector,
        x_vector=initial_x,
        initial_alpha=initial_alpha,
        rho=0.5,
        c=0.1,
        )
    else:
        alpha = backtracking(
            func,
            gradient_function,
            p_vector=p_vector,
            x_vector=initial_x,
            initial_alpha=initial_alpha,
            rho=0.9,
            c=0.1,
        )
    new_x = initial_x + (alpha * p_vector)
    
    difference = np.linalg.norm(initial_x - new_x)
    convergence.append(difference)
    print(iteration, difference)
    if (difference < delta) or (iteration >= maximum_interation):
        return new_x

    else:
        
        new_grad = gradient_function(new_x)
        S_value = new_x - initial_x
        Y_value = new_grad - grad
        
        denom1 = (Y_value.T @ S_value) ** 2
        denom2 = Y_value.T @ S_value

        F_value_change1 = ((Y_value.T @ ((F_value @ Y_value) + S_value)) / denom1) * (
            S_value @ S_value.T
        )
        F_value_change2 = (
            (S_value @ (Y_value.T @ F_value)) + (F_value @ Y_value @ S_value.T)
        ) / denom2

        new_F_value = F_value + F_value_change1 - F_value_change2

        return quasi_newton(
            func,
            gradient_function,
            F_value=new_F_value,
            initial_x=new_x,
            initial_alpha=initial_alpha,
            delta=delta,
            maximum_interation=maximum_interation,
            iteration=iteration,
            convergence=convergence,
            constraint=constraint
        )


if __name__ == "__main__":
    
    optimum_x_value1 = quasi_newton(
        func_1,
        actual_gradient_func1,
        F_value=np.array([[1, 0], [0, 1]]),
        initial_x=np.array([50, 50]).reshape(2,1),
        initial_alpha=0.1,
        delta=1e-6,
        maximum_interation=1000,
        iteration=1,
        convergence=[],
        constraint=False
    )
    print(optimum_x_value1)
