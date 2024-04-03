import numpy as np

def leapfrog_1d(dVdx, x, v, h, N):
    """
    dVdx - gradient of the potential energy wrt position (parameter), a function
    x - position (float)
    v - velocity, (float)
    h - step length, (float)
    N - number of steps, integer
    """

    trace_v = [v]
    trace_x = [x]

    # start with half step, then full steps in loop, finally half step.
    v = v - (h/2) * dVdx(x)
    
    for _ in range(N-1): 
        x = x + h * v 
        v = v - h * dVdx(x)  
        trace_v.append(v)
        trace_x.append(x)

    x = x + h * v
    v = v - (h/2) * dVdx(x)
    
    trace_v.append(v)
    trace_x.append(x)

    trace_v, trace_x = np.array(trace_v), np.array(trace_x)

    return x, v, trace_x, trace_v

