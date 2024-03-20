
import numpy as np

class GaussianMixtureND:

    def __init__(self, mus, sigmas):
        assert len(mus) == len(sigmas), "mus and sigmas must have the same length"
        self.mus = mus  
        self.sigmas = sigmas 
        self.n_modes = len(self.mus)
        
        self.weights = [1.0 / self.n_modes] * self.n_modes  # Assuming equal weights for simplicity

    def normal_distribution(self, x, mu, sigma):
        d = len(mu)
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        norm_const = 1.0 / (np.power((2 * np.pi), float(d) / 2) * np.sqrt(sigma_det))
        x_mu = np.array(x - mu)
        result = np.exp(-0.5 * (x_mu.T.dot(sigma_inv).dot(x_mu)))
        return norm_const * result

    def pdf(self, x):
        return sum(weight * self.normal_distribution(x, mu, sigma) 
                   for weight, mu, sigma in zip(self.weights, self.mus, self.sigmas))

    def log_pdf(self, x):
        return np.log(self.pdf(x))

    def gradient_log_pdf(self, x):
        if self.n_modes == 1:
            sigma_inv = np.linalg.inv(self.sigmas[0])
            return -sigma_inv.dot(x - self.mus[0])
        else:
            numerator = sum(weight * self.normal_distribution(x, mu, sigma) * -np.linalg.inv(sigma).dot(x - mu)
                            for weight, mu, sigma in zip(self.weights, self.mus, self.sigmas))
            denominator = self.pdf(x)
            return numerator / denominator
        
    def dVdx(self, x):
        return -self.gradient_log_pdf(x)
    

    def marginalized_pdf(self, x, dims):
        """
        Compute the marginalized PDF over the specified dimensions.
        """
        marginal_mus = [mu[dims] for mu in self.mus]
        marginal_sigmas = [sigma[np.ix_(dims, dims)] for sigma in self.sigmas]

        return sum(weight * self.normal_distribution(x, mu, sigma) 
                   for weight, mu, sigma in zip(self.weights, marginal_mus, marginal_sigmas))
