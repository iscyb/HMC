import numpy as np

class GaussianMixture1d:

    def __init__(self, mu, std):
        assert len(mu) == len(std), "mu and std must have the same length"
        self.mu = mu
        self.std = std
        self.n_modes = len(self.mu)
        
        self.weights = [1.0 / self.n_modes] * self.n_modes # Assuming equal weights for simplicity

    def normal_distribution(self, x, mu, std):
        return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-0.5 * ((x - mu) / std) ** 2)

    def pdf(self, x):
        out = sum(weight * self.normal_distribution(x, mu, std) for weight, mu, std in zip(self.weights, self.mu, self.std))
        return out

    def log_pdf(self, x):
        return np.log(self.pdf(x))

    def gradient_log_pdf(self, x):
        if self.n_modes == 1:
            return -(x - self.mu[0]) / self.std[0] ** 2
        else:
            numerator = sum(weight * self.normal_distribution(x, mu, std) * (-(x - mu) / std ** 2)
                            for weight, mu, std in zip(self.weights, self.mu, self.std))
            denominator = self.pdf(x)
            return numerator / denominator
        
    def dVdx(self, x):
        return -self.gradient_log_pdf(x)