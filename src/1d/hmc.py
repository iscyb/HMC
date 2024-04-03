
import numpy as np

from integrators import leapfrog_1d


def hmc_1d(target_distribution, initial_state, iterations=10_000, step_size=0.1, n_steps=10, integrator='leapfrog'):
    samples = [initial_state]
    accept_or_not = []
    trace_x_all = [] 
    trace_v_all = [] 

    dVdx = target_distribution.dVdx

    for i in range(iterations):
        if i%1000 == 0:
            print(f'Current iteration: {i}')

        x0 = samples[-1]
        v0 = np.random.normal()

        if integrator == 'leapfrog':
            x_star, v_star, trace_x, trace_v = leapfrog_1d(dVdx, x0, v0, step_size, n_steps)

        # Metropolis acceptance step
        m0 = -target_distribution.log_pdf(x0) + 0.5 * v0**2
        m_star = -target_distribution.log_pdf(x_star) + 0.5 * v_star**2
        accept_prob = np.exp(m0 - m_star)

        if np.random.uniform() < accept_prob:
            samples.append(x_star)
            accept_or_not.append(True) 
            trace_x_all.append(trace_x)
            trace_v_all.append(trace_v)
        else:
            samples.append(x0)
            accept_or_not.append(False)

    return samples, trace_x_all, trace_v_all, accept_or_not
