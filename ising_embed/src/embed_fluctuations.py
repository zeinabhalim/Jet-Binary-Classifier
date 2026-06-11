#!/usr/bin/env python3
"""
embed_fluctuations.py — Imprint 3D Ising critical fluctuations
on a vHLLE freezeout surface.

Uses qcd_mapping.IsingToQCDMap and ising_scaling.IsingScalingFunctions.

Usage:
  python3 embed_fluctuations.py freezeout_7gev.dat -o modified.dat --strength 0.2 --info
"""

import sys, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freezeout_reader import read_freezeout, write_freezeout, FreezeoutSurface
from qcd_mapping import IsingToQCDMap
from ising_scaling import IsingScalingFunctions


def ising_magnetization(r, h, beta=0.326, gamma=1.237, delta=4.80):
    """M from (r, h) via leading-order power laws."""
    M = np.zeros_like(r)
    eps = 1e-15
    neg = r < -eps
    pos = r > eps
    zero = ~(neg | pos)
    M[neg] = np.sign(h[neg] + eps) * np.abs(r[neg]) ** beta
    M[pos] = h[pos] / (r[pos] ** gamma + eps)
    M[zero] = np.sign(h[zero] + eps) * (np.abs(h[zero]) + eps) ** (1.0 / delta)
    return np.clip(M, -1.0, 1.0)


def embed_surface(surface, strength=0.2, T_cp_MeV=143.2, muB_cp_MeV=350.0,
                  alpha1=3.85, alpha2=93.85, rho=2.0, w=1.0,
                  delta_T_MeV=30.0, delta_muB_MeV=80.0, seed=42):
    rng = np.random.default_rng(seed)
    qcd_map = IsingToQCDMap(Tc=T_cp_MeV, muBc=muB_cp_MeV,
                            alpha1=alpha1, alpha2=alpha2, rho=rho, w=w)

    T_MeV = surface.T * 1000.0        # GeV → MeV
    muB_MeV = surface.mu_B * 1000.0   # GeV → MeV

    r, h = qcd_map.qcd_to_ising(T_MeV, muB_MeV)
    M = ising_magnetization(r, h)

    w_env = np.exp(
        -((T_MeV - T_cp_MeV) / delta_T_MeV)**2
        - ((muB_MeV - muB_cp_MeV) / delta_muB_MeV)**2
    )
    delta_mu = strength * w_env * M * rng.choice([-1, 1], size=surface.n_cells)

    return FreezeoutSurface(
        tau=surface.tau.copy(), x=surface.x.copy(),
        y=surface.y.copy(), eta_s=surface.eta_s.copy(),
        dsigma=surface.dsigma.copy(), u=surface.u.copy(),
        T=surface.T.copy(),
        mu_B=surface.mu_B + delta_mu,
        mu_Q=surface.mu_Q.copy(), mu_S=surface.mu_S.copy(),
        shear=surface.shear.copy(), flag=surface.flag.copy(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default="modified_freezeout.dat")
    p.add_argument("--strength", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--info", action="store_true")
    args = p.parse_args()

    print(f"Reading: {args.input}")
    surface = read_freezeout(args.input)
    print(surface.summary())

    modified = embed_surface(surface, strength=args.strength, seed=args.seed)
    write_freezeout(modified, args.output)
    print(f"Written: {args.output}")

    if args.info:
        delta = modified.mu_B - surface.mu_B
        print(f"  |delta_muB| max:  {np.abs(delta).max():.6f} GeV")
        print(f"  |delta_muB| mean: {np.abs(delta).mean():.6f} GeV")
        T_cp, muB_cp = 143.2, 350.0
        n_near = ((surface.T*1000 > T_cp - 20) & (surface.T*1000 < T_cp + 20)
                  & (surface.mu_B*1000 > muB_cp - 50)
                  & (surface.mu_B*1000 < muB_cp + 50)).sum()
        print(f"  Cells near CP:    {n_near}")


if __name__ == "__main__":
    main()
