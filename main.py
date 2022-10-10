import numpy as np
from quasi_newton import quasi_newton
from cg import cg
from gradient_descent import gradient_descent
from functions import (
    problem_function_1,
    gradient_function_1,
    problem_function_3,
    gradient_function_3,
    problem_function_2,
    gradient_function_2
)

import matplotlib.pyplot as plt

# ....

def optimize():
    convergence1 = []

    quasi_newton_fuction_1 = quasi_newton(
        problem_function_1,
        gradient_function_1,
        F_value=np.identity(100),
        initial_x=20 * np.ones(100).reshape(-1, 1),
        initial_alpha=1,
        delta=1e-6,
        maximum_interation=1000,
        iteration=1,
        convergence = convergence1,
        constraint=False
    )
    print(problem_function_1(quasi_newton_fuction_1))

    res = cg(func=problem_function_1, x=20 * np.ones(100).reshape(-1, 1))

    distance = np.array([(np.linalg.norm(np.array(x).reshape(-1,1) - np.zeros(100).reshape(-1,1))) for x in res.allvecs])
    plt.plot(range(1, len(res.allvecs)+1), distance, marker='o', markerfacecolor='green', markersize=4)
    plt.xlabel('Iteration i')
    plt.ylabel('L2 Norm with of x(i) respect to true value')
    plt.title('Conjugate Gradient for function 1')
    plt.show()

    print(problem_function_1(res.x))

    distance = np.array([(np.linalg.norm(np.array(x).reshape(-1,1) - np.zeros(100).reshape(-1,1))) for x in convergence1])
    plt.plot(range(1, len(convergence1)+1), distance, marker='o', markerfacecolor='green', markersize=4)
    plt.xlabel('Iteration i')
    plt.ylabel('L2 Norm with of x(i) respect to true value')
    plt.title('Quasi Newton for function 1')
    plt.show()



    convergence2 = []
    quasi_newton_fuction_2 = quasi_newton(
        problem_function_2,
        gradient_function_2,
        F_value= np.identity(100),
        initial_x=  np.zeros(100).reshape(-1, 1),
        initial_alpha=1.0,
        delta=1e-3,
        maximum_interation=1000,
        iteration=1,
        convergence = convergence2,
        constraint=True
    )
    print(problem_function_2(quasi_newton_fuction_2))

    plt.plot(range(1, len(convergence2)+1), convergence2, marker='o', markerfacecolor='green', markersize=4)
    plt.xlabel('Iteration i')
    plt.ylabel('L2 Norm with of x(i) respect to true value')
    plt.title('Quasi Newton for function 2')
    plt.show()
    
    convergence3 = []
    res = cg(func=problem_function_2, x=np.zeros(100).reshape(-1, 1))

    print(problem_function_2(res.x))

    quasi_newton_fuction_3 = quasi_newton(
        problem_function_3,
        gradient_function_3,
        F_value= np.identity(2),
        initial_x=  np.array([50,100]).reshape(-1, 1),
        initial_alpha=1,
        delta=1e-6,
        maximum_interation=1000,
        iteration=1,
        convergence=convergence3,
        constraint=False
    )
    print(problem_function_3(quasi_newton_fuction_3))
    distance = np.array([(np.linalg.norm(x - [1,1])) for x in convergence3])
    plt.plot(range(1, len(convergence3)+1), distance, marker='o', markerfacecolor='green', markersize=4)
    plt.xlabel('Iteration i')
    plt.ylabel('L2 Norm with of x(i) respect to true value')
    plt.title('Quasi Newton for function 3')
    plt.show()

    res = cg(func=problem_function_3, x=np.array([50,100]).reshape(-1, 1))

    distance = np.array([(np.linalg.norm(x - [1,1])) for x in res.allvecs])
    plt.plot(range(1, len(res.allvecs)+1), distance, marker='o', markerfacecolor='green', markersize=4)
    plt.xlabel('Iteration i')
    plt.ylabel('L2 Norm with of x(i) respect to true value')
    plt.title('Conjugate Gradient for function 3')
    plt.show()


if __name__ == "__main__":
    optimize()
