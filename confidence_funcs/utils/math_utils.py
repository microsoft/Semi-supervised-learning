from scipy.optimize import minimize 
from math import cos 
import numpy as np 

def get_vector_at_angle(theta,w):
    d = len(w)
    def fun(u):
        return abs(np.dot(u,w)-cos(theta )) 
    def const_fun(u):
        return abs(np.linalg.norm(u)-1)

    consts = {"type":'eq', 'fun':const_fun}
    u0 = np.zeros(d)
    u0[0]= 1 #np.random.uniform(d)
    out = minimize(fun,u0,tol=1e-10,constraints=consts)
    u = out['x']
    return u 

