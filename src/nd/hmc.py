import numpy as np

from integrators import leapfrog, euler, split_step

# TODO unify with 1d.

def hmc_nd(target_distribution, initial_state, iterations=10_000, step_size=0.1, n_steps=10, integrator='leapfrog'):
    """
    Parameters:
    - target_distribution: An object with .log_pdf(x) and .dVdx(x) methods.
    - initial_state: A numpy array representing the initial state in the parameter space.
    - iterations: Number of iterations to run the HMC sampler.
    - step_size: Step size (epsilon) for the leapfrog integrator.
    - n_steps: Number of leapfrog steps per HMC iteration.
    """
    samples = [initial_state]
    accept_or_not = []
    trace_x_all = []  
    trace_v_all = [] 

    dVdx = target_distribution.dVdx  # gradient of the -log PDF

    for i in range(iterations):
        if i % 1000 == 0:
            print(f'Current iteration: {i}')

        x0 = samples[-1]
        v0 = np.random.normal(size=initial_state.shape) 

        # Perform the leapfrog integration
        if integrator == 'leapfrog':
            trace_x, trace_v, x_star, v_star = leapfrog(dVdx, x0, v0, step_size, n_steps)
        elif integrator == 'euler':
            trace_x, trace_v, x_star, v_star = euler(dVdx, x0, v0, step_size, n_steps)
        elif integrator == 'splitstep':
            trace_x, trace_v, x_star, v_star = split_step(dVdx, x0, v0, step_size, n_steps)

        # Metropolis step
        m0 = -target_distribution.log_pdf(x0) + 0.5 * np.sum(v0**2)
        m_star = -target_distribution.log_pdf(x_star) + 0.5 * np.sum(v_star**2)
        accept_prob = np.exp(m0 - m_star)

        if np.random.uniform() < accept_prob:
            samples.append(x_star)
            accept_or_not.append(True) 
            trace_x_all.append(trace_x)
            trace_v_all.append(trace_v)
        else:
            samples.append(x0)
            accept_or_not.append(False) 

    samples = np.array(samples)
    return samples, trace_x_all, trace_v_all, accept_or_not