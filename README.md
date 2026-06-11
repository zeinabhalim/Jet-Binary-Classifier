# QCD Matter Tomography: Jet Classification & Critical Point Detection

 Quark- and gluon-initiated jets exhibit subtle differences arising from Quantum Chromodynamics (QCD) color factors. As the gluons carry a larger color charge ( \( C_A = 3 \) ) than quarks ( \( C_F = 4/3 \) ) therefore gluon jets radiate more intensely. This enhanced radiation leads to broader jets with higher particle multiplicity and variations in jet substructure observables such as:

- Particle multiplicity (charged and neutral constituents)
- Jet mass
- N-subjettiness ratios (τ21)
- Jet width (Girth)
- jet eccentricity


 A dual-pipeline framework for probing QCD matter structure developed in two stages:

- **Stage 1 — Jet Binary Classifier**: Quark vs. gluon jet discrimination in pp collisions at √s = 13 TeV using PYTHIA8 + FastJet + PyTorch
  
Accordingly, the combined correlations among these observables allow statistical discrimination between quark and gluon jets that implemented in this stage as follows;

1. **Monte Carlo Simulation**  
   Proton–proton (pp) collisions at √s = 13 TeV are generated using **PYTHIA8** event generation.

2. **Jet Clustering**  
   Final-state particles are clustered into jets using the anti-kₜ algorithm implemented in **FastJet**.

3. **Feature Extraction**  
   For each reconstructed jet, aforementioned substructure observables are computed and stored.

4. **Machine Learning Classification**  
   Extracted jet features are used to train a fully connected neural network implemented in **PyTorch**, optimized using:
   - Binary Cross Entropy (BCE) loss  
   - Adam optimizer
   - The results had evaluated using the ROC/AUC ratio curve.
     
- **Stage 2 — Ising Embed**: Detection of the QCD critical point signature in Au+Au collisions at RHIC BES-II energies (√s = 7.7–19.6 GeV) using the hybrid hydrodynamics framework with 3D Ising universality class embedding

The QCD phase diagram has a conjectured second-order critical point $(T_c, \mu_{B,c})$ separating the smooth crossover region (high $T$, low $\mu_B$) from a first-order phase transition line. Near this point, the system belongs to the **3D Ising universality class** — the identical universality class as a uniaxial ferromagnet near its Curie point. This is exact: the long-wavelength fluctuations depend only on the $\mathbb{Z}_2$ symmetry and spatial dimension $d=3$, not on the microscopic QCD Lagrangian.

The net-baryon number cumulants diverge as powers of the correlation length $\xi$ (Stephanov 2011):

$$\kappa_2 \sim \xi^{2-\eta} \approx \xi^{1.96}, \quad \kappa_3 \sim \xi^{\beta\delta + \gamma/2} \approx \xi^{4.5}, \quad \kappa_4 \sim \xi^{2\beta\delta + \gamma} \approx \xi^{7}$$

where $\beta=0.326$, $\delta=4.80$, $\gamma=1.237$, $\eta=0.036$ are the 3D Ising critical exponents. The kurtosis ratio $\kappa_4/\kappa_2 \sim \xi^{5}$ is the volume-independent and maximally sensitive to the critical point.

The RHIC Beam Energy Scan probes $\mu_B \in [112, 420]$ MeV across $\sqrt{s_{NN}} = 7.7$–62.4 GeV. At 7.7 GeV, $\langle\mu_B\rangle \approx 421$ MeV — closest to the hypothesized critical point location.

### Theoretical Pipeline
**Step 1 — Linear field mapping** (Parotto et al., Phys. Rev. C 101, 034901, 2020):

The QCD thermodynamic coordinates $(T, \mu_B)$ map to the Ising scaling fields $(r, h)$ via:

$$r = \frac{(T - T_c)\cos\alpha_2/T_c + (\mu_B - \mu_{Bc})\sin\alpha_2/T_c}{w\rho(\sin\alpha_1\cos\alpha_2 - \cos\alpha_1\sin\alpha_2)}$$

$$h = \frac{-(T - T_c)\cos\alpha_1/T_c - (\mu_B - \mu_{Bc})\sin\alpha_1/T_c}{-w(\sin\alpha_1\cos\alpha_2 - \cos\alpha_1\sin\alpha_2)}$$

Parameters: $T_c = 143.2$ MeV, $\mu_{Bc} = 350$ MeV, $\alpha_1 = 3.85°$, $\alpha_2 = 93.85°$, $w = 1.0$, $\rho = 2.0$.

**Step 2 — Correlation length** (Schofield parametrization, 3D Ising universality class):

$$\xi = \xi_0 \left(r^2 + |h|^{2/\delta}\right)^{-\nu/2}, \quad \nu = 0.630,\; \delta = 4.80,\; \xi_{\rm max} = 3 \text{ fm (non-equilibrium cap)}$$

**Step 3 — Cumulant enhancement** per freeze-out cell, with embedding strength $s$:

$$\kappa_n^{\rm cell} = 1 + s\,\left(\xi^{\,\text{exponent}_n} - 1\right)$$

**Step 4 — Cooper-Frye particlization** with enhanced fluctuation weights the final-state hadrons.


### Run Cooper-Frye Sampler

```bash
SAMPLER=~/Documents/vhlle-smash/smash-hadron-sampler_v3.0_backup/build/sampler

# Correct calling convention: events N config_file
$SAMPLER events 1000 configs/sampler_7gev.txt
$SAMPLER events 1000 configs/sampler_11gev.txt
$SAMPLER events 1000 configs/sampler_19gev.txt
```

Each config file specifies `surface`, `spectra_dir`, `number_of_events`, `shear`, `ecrit`.

## Repository Structure

```
JetBinaryClass/
│
├── CMakeLists.txt                    # Build: PYTHIA8 + FastJet 
├── LICENSE
├── README.md
│
├── src/                              # Stage 1: pp jet generation 
│   └── generate_pythia_jets.cc       # PYTHIA8 event gen + FastJet clustering
│                                     # Output: jet substructure CSV
│
├── MLmodel/                          # Stage 1: PyTorch binary classifier
│   └── train.py                      # BCE loss · Adam · ROC-AUC evaluation
│
├── results/                          # ROC curves, AUC scores, feature plots
│
└── ising_embed/                      # Stage 2: QCD critical point pipeline
    │
    ├── parameters.yaml               # All physics parameters (Tc, μBc, w, ρ, α1, α2)
    │                                 # Ising exponents, BES run config, ML features
    │
    ├── src/
    │   ├── qcd_mapping.py            # (T, μB) → (r, h) linear Ising field map
    │   │                             # Parotto et al. 2020: Tc=143.2 MeV, μBc=350 MeV
    │   ├── ising_scaling.py          # Schofield parametrization + κ₂, κ₃, κ₄ scaling
    │   │                             # 3D Ising exponents: ν=0.630, δ=4.80, η=0.036
    │   ├── freezeout_reader.py       # vHLLE freeze-out surface parser (27-column format)
    │   └── embed_fluctuations.py     # Imprint Ising fluctuations per freeze-out cell
    │
    ├── analysis/
    │   ├── cluster_heavyion_jets.cc      # FastJet clustering on SMASH particle lists
    │   ├── train_heavyion.py             # PyTorch: signal (7.7 GeV) vs baseline (19.6 GeV)
    │   ├── jets_signal_7gev.csv          # Jet features — signal,   √s = 7.7  GeV (label=1)
    │   ├── jets_control_11gev.csv        # Jet features — control,  √s = 11.5 GeV (label=1)
    │   ├── jets_baseline_19gev.csv       # Jet features — baseline, √s = 19.6 GeV (label=0)
    │   ├── jets_training.csv             # Combined training set (signal + baseline)
    │   ├── BES_critical_point_signal.png # κ₄/κ₂ vs √s — main physics result
    │   └── heavyion_classifier_results.png  # ROC curves across BES energies
    │
    └── data/
        ├── signal_freezeout_7gev.dat     # Ising-embedded freeze-out, √s = 7.7  GeV
        ├── control_11gev.dat             # Ising-embedded freeze-out, √s = 11.5 GeV
        ├── baseline_19gev.dat            # Ising-embedded freeze-out, √s = 19.6 GeV
        │
        └── vHlle-sampler BES II/         # Cooper-Frye sampled particle lists
            ├── signal_7gev/
            │   ├── particle_lists.oscar  # 1000 hadronic events, √s = 7.7  GeV
            │   └── 1000.root
            ├── control_11gev/
            │   ├── particle_lists.oscar  # 1000 hadronic events, √s = 11.5 GeV
            │   └── 1000.root
            └── baseline_19gev/
                ├── particle_lists.oscar  # 1000 hadronic events, √s = 19.6 GeV
                └── 1000.root
```

---


## Software Dependencies

| Package | Version | Purpose |
|---|---|---|
| PYTHIA8 | 8.309 / 8.316 | pp event generation |
| FastJet | ≥ 3.4 | Anti-kT jet clustering |
| PyTorch | ≥ 2.0 | Neural network training |
| vHLLE | latest | Viscous 3+1D relativistic hydro |
| SMASH-hadron-sampler | v3.0 | Cooper-Frye particlization |
| NumPy / SciPy | latest | Numerical analysis |
| CMake | ≥ 3.15 | C++ build system |
