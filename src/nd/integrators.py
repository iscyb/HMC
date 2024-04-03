import numpy as np

# TODO Unify with 1d

def leapfrog(dVdx, x, v, h, N, M=None):
    """
    dVdx - gradient of the potential energy wrt position (parameter), a function
    x - position, array
    v - velocity, array
    h - step length
    N - number of steps
    M - mass matrix (optional; if None, identity is assumed)
    """
    
    x = np.asarray(x)
    v = np.asarray(v)
    
    if M is None:
        M_inv = np.identity(len(x)) if x.ndim > 0 else 1 
    else:
        M_inv = np.linalg.inv(M) 
    
    trace_x = [x.copy()]
    trace_v = [v.copy()]

    v = v - (h / 2) * dVdx(x)
    
    for _ in range(N-1):
        x = x + h * np.dot(M_inv, v)  
        v = v - h * dVdx(x)           
        trace_x.append(x.copy())
        trace_v.append(v.copy())

    x = x + h * np.dot(M_inv, v)
    v = v - (h / 2) * dVdx(x) 

    trace_x.append(x.copy())
    trace_v.append(v.copy())

    return np.array(trace_x), np.array(trace_v), x, v

def euler(dVdx, x, v, h, N, M=None):
    """
    dVdx - gradient of the potential energy wrt position (parameter), a function
    x - position, array
    v - velocity, array
    h - step length
    N - number of steps
    M - mass matrix (optional; if None, identity is assumed)
    """
    x = np.asarray(x)
    v = np.asarray(v)
    
    if M is None:
        M_inv = np.identity(len(x)) if x.ndim > 0 else 1  
    else:
        M_inv = np.linalg.inv(M)
    
    trace_x = [x.copy()]
    trace_v = [v.copy()]

    for _ in range(N):
        x = x + h * np.dot(M_inv, v)  
        v = v - h * dVdx(x)         
        trace_x.append(x.copy())
        trace_v.append(v.copy())

    return np.array(trace_x), np.array(trace_v), x, v


def split_step(dVdx, x, v, h, N, b=0.2118, M=None):
    """
    dVdx - gradient of the potential energy wrt position (parameter), a function
    x - position, array
    v - velocity, array
    h - step length
    N - number of steps
    M - mass matrix (optional; if None, identity is assumed)
    b - Parameter defining the fraction of the step for velocity updates in the split-step method.
        Important values: minimal error: 0.1932. An alernative is b = 0.2118 : Source Sanz Serna
    M - mass matrix (optional; if None, identity is assumed)
    """
    x = np.asarray(x)
    v = np.asarray(v)
    
    if M is None:
        M_inv = np.identity(len(x)) if x.ndim > 0 else 1  
    else:
        M_inv = np.linalg.inv(M)
    
    trace_x = [x.copy()]
    trace_v = [v.copy()]

    for _ in range(N):
        v = v - b * h * dVdx(x)
        x = x + (h / 2) * np.dot(M_inv, v)
        v = v - (1 - 2*b) * h * dVdx(x)
        x = x + (h / 2) * np.dot(M_inv, v)
        v = v - b * h * dVdx(x)
        
        trace_x.append(x.copy())
        trace_v.append(v.copy())

    return np.array(trace_x), np.array(trace_v), x, v
