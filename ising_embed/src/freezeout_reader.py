#!/usr/bin/env python3
"""
freezeout_reader.py — Parse vHLLE freezeout surface files.

27-column layout (0-indexed):
   0-3:   (tau, x, y, eta_s)     — space-time
   4-7:   dsigma_mu               — hypersurface normal
   8-11:  u_mu (γ, γvx, γvy, γvz) — flow 4-velocity
   12:    T [GeV]                 — temperature
   13:    mu_B [GeV]              — baryon chem. pot.
   14:    mu_Q [GeV]              — charge chem. pot.
   15:    mu_S [GeV]              — strangeness chem. pot.
   16-25: pi^{mu nu}              — shear stress
   26:    flag                    — freezeout flag
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FreezeoutSurface:
    tau:    np.ndarray
    x:      np.ndarray
    y:      np.ndarray
    eta_s:  np.ndarray
    dsigma: np.ndarray   # (N, 4)
    u:      np.ndarray   # (N, 4)
    T:      np.ndarray
    mu_B:   np.ndarray
    mu_Q:   np.ndarray
    mu_S:   np.ndarray
    shear:  np.ndarray   # (N, 10)
    flag:   np.ndarray

    @property
    def n_cells(self) -> int:
        return len(self.T)

    def summary(self) -> str:
        return (
            f"Cells:   {self.n_cells}\n"
            f"T:       {self.T.min():.4f} - {self.T.max():.4f} GeV  (mean {self.T.mean():.4f})\n"
            f"mu_B:    {self.mu_B.min():.4f} - {self.mu_B.max():.4f} GeV  (mean {self.mu_B.mean():.4f})\n"
            f"tau:     {self.tau.min():.1f} - {self.tau.max():.1f} fm\n"
            f"eta_s:   {self.eta_s.min():.2f} - {self.eta_s.max():.2f}"
        )


def read_freezeout(path: str | Path) -> FreezeoutSurface:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Freezeout file not found: {path}")
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != 27:
        raise ValueError(f"Expected 27 columns, got {raw.shape[1]}")

    return FreezeoutSurface(
        tau=raw[:, 0], x=raw[:, 1], y=raw[:, 2], eta_s=raw[:, 3],
        dsigma=raw[:, 4:8], u=raw[:, 8:12],
        T=raw[:, 12], mu_B=raw[:, 13], mu_Q=raw[:, 14], mu_S=raw[:, 15],
        shear=raw[:, 16:26], flag=raw[:, 26],
    )


def write_freezeout(surface: FreezeoutSurface, path: str | Path) -> None:
    cols = [surface.tau, surface.x, surface.y, surface.eta_s]
    cols.extend([surface.dsigma[:, i] for i in range(4)])
    cols.extend([surface.u[:, i] for i in range(4)])
    cols += [surface.T, surface.mu_B, surface.mu_Q, surface.mu_S]
    cols.extend([surface.shear[:, i] for i in range(10)])
    cols.append(surface.flag)
    np.savetxt(path, np.column_stack(cols), fmt="%.12e")

