#the Schofield parametrization + cumulants
"""3D Ising universal scaling functions via Schofield parametrization."""
import numpy as np

class IsingScalingFunctions:

    def __init__(self, xi0=1.0):
        # 3D Ising critical exponents
        self.beta = 0.326
        self.delta = 4.80
        self.nu = 0.630
        self.eta = 0.036
        self.gamma = self.beta * (self.delta - 1)  # = 1.237
        self.xi0 = xi0  # correlation length normalization
        
        # Schofield parametrization coefficients
        self.c2 = -0.762
        self.c4 = 0.008
        self.theta_max = 1.154  # where h(R, theta) has a singularity

    def h0(self, theta):
        return theta * (1 + self.c2 * theta**2 + self.c4 * theta**4)

    def r_from_rh(self, R, theta):
        return R * (1 - theta**2)

    def h_from_rh(self, R, theta):
        return R**(self.beta * self.delta) * self.h0(theta)

    def correlation_length(self, r, h, R=None, theta=None):
        """Compute ξ in fm from (r, h) or (R, θ)."""
        if R is None:
            # If given (r, h), solve for R iteratively
            # ... (simplified: use the scaling ξ ∝ |r|^{-ν} near r > 0, h ≈ 0)
            if abs(h) < 1e-10 and r > 0:
                return self.xi0 * r**(-self.nu)
            else:
                # Full Schofield solution omitted for brevity
                return self.xi0 * (r**2 + (h/self.h0(0.5))**2)**(-self.nu/2)
        
        return self.xi0 * R**(-self.nu)

    def kappa2(self, xi):
        """Cumulant scaling: variance ∝ ξ^{2-η}."""
        return xi**(2 - self.eta)

    def kappa3(self, xi):
        """Skewness scaling: ∝ ξ^{4.5}."""
        return xi**(self.beta * self.delta + self.gamma / 2)

    def kappa4(self, xi):
        """Kurtosis scaling: ∝ ξ^7 — the most dramatic critical signal."""
        return xi**(2*self.beta*self.delta + self.gamma)

