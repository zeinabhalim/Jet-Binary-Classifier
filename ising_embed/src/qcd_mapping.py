#the linear map inspired by ref. Parotto et al. 2020
import numpy as np

class IsingToQCDMap:
    def __init__(self, Tc=143.2, muBc=350.0, alpha1=3.85,
                 alpha2=93.85, rho=2.0, w=1.0):
        self.Tc = Tc
        self.muBc = muBc
        self.alpha1 = np.radians(alpha1)
        self.alpha2 = np.radians(alpha2)
        self.rho = rho
        self.w = w
        # Precompute denominator for the inverse map
        denom = w * rho * (np.sin(self.alpha1)*np.cos(self.alpha2)
                           - np.cos(self.alpha1)*np.sin(self.alpha2))
        self.denom = denom
        self.denom_h = -w * (np.sin(self.alpha1)*np.cos(self.alpha2)
                              - np.cos(self.alpha1)*np.sin(self.alpha2))

    def qcd_to_ising(self, T, muB):
        """Map (T, muB) → (r, h) for each freeze-out point cell."""
        dT = (T - self.Tc) / self.Tc
        dmu = (muB - self.muBc) / self.Tc
        r = (dT * np.cos(self.alpha2) + dmu * np.sin(self.alpha2)) / self.denom
        h = (-dT * np.cos(self.alpha1) - dmu * np.sin(self.alpha1)) / self.denom_h
        return r, h

