import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from functions import func_1, get_a_b_c_coefficient

def cons_fx(x):
    A, B, C = get_a_b_c_coefficient()
    constraint_value = B - A @ x.reshape(-1,1) 
    return constraint_value

def cg(func, x):
    
    constraints = NonlinearConstraint(fun = cons_fx, lb = np.zeros(500).reshape(-1,1), ub=None)
    res = minimize(func, x, method='CG',
               options={'gtol': 1e-03, 'disp': True, 'return_all': True, 'maxiter' : 2}, constraints=[constraints])

    return res


if __name__ == "__main__":
    print(cg(func_1, np.array([50,50]).reshape(-1,1)).x)
    
