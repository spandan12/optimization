import numpy as np
from quasi_newton import quasi_newton
from functions import problem_function_1, gradient_function_1

def optimize():
    quasi_newton_fuction_1 = quasi_newton(
            problem_function_1,
            gradient_function_1,
            F_value=np.identity(100),
            initial_x=50*np.ones(100).reshape(-1,1),
            initial_alpha=1,
            delta=1e-6,
            maximum_interation=1000,
            iteration=1,
        )
    print(quasi_newton_fuction_1)

if __name__ == "__main__":
    optimize()
