 As the quark- and gluon-initiated jets exhibit subtle differences arising from Quantum Chromodynamics (QCD) color factors. Since gluons carry a larger color charge ($C_A = 3$) than quarks ($C_F = 4/3$), gluon jets radiate more intensely, producing broader jets with higher particle multiplicity and variations in jet substructure observables such as:
 
- Particle multiplicity (charged and neutral constituents)
- Jet mass
- N-subjettiness ratios ($\tau_{21}$)
- Jet width (Girth)
- Jet eccentricity
  
The project constructs a dual-pipeline framework for probing QGP dynamics, developed in two stages:
 
- **Stage 1 — Jet Binary Classifier**: Quark vs. gluon jet discrimination in pp collisions at $\sqrt{s} = 13$ TeV using PYTHIA8 + FastJet + PyTorch
- **Stage 2 — Ising Embed**: Detection of the QCD critical point signature in Au+Au collisions at RHIC BES-II energies ($\sqrt{s_{NN}} = 7.7$–$19.6$ GeV) using a hybrid hydrodynamics framework with 3D Ising universality class embedding
---
 
## Stage 1 — p-p Jet Binary Classifier
 
The combined correlations among jet substructure observables allow statistical discrimination between quark- and gluon-initiated jets, implemented as follows:
 
1. **Monte Carlo Simulation** — Proton-proton collisions at $\sqrt{s} = 13$ TeV generated using **PYTHIA8**
2. **Jet Clustering** — Final-state particles clustered into jets using the anti-$k_T$ algorithm implemented in **FastJet**
3. **Feature Extraction** — Jet substructure observables (multiplicity, jet mass, $\tau_{21}$, girth, eccentricity) computed per reconstructed jet
4. **Machine Learning Classification** — A fully connected neural network trained in **PyTorch** using:
   - Binary Cross Entropy (BCE) loss
   - Adam optimizer
   - ROC-AUC evaluation
---
 
## Stage 2 — QCD Critical Point Detection
 
The QCD phase diagram has a conjectured second-order critical point $(T_c, \mu_{B,c})$ separating the smooth crossover region (high $T$, low $\mu_B$) from a first-order phase transition line. Near this point, the system belongs to the **3D Ising universality class**. The net-baryon number cumulants diverge as powers of the correlation length $\xi$ (Stephanov 2011):
 
$$\kappa_2 \sim \xi^{2-\eta} \approx \xi^{1.96}, \quad \kappa_3 \sim \xi^{\beta\delta + \gamma/2} \approx \xi^{4.5}, \quad \kappa_4 \sim \xi^{2\beta\delta + \gamma} \approx \xi^{7}$$
 
where $\beta=0.326$, $\delta=4.80$, $\gamma=1.237$, $\eta=0.036$ are the 3D Ising critical exponents. The kurtosis ratio $\kappa_4/\kappa_2 \sim \xi^{5}$ is volume-independent and maximally sensitive to the critical point.
 
The RHIC Beam Energy Scan probes $\mu_B \in [112, 420]$ MeV across $\sqrt{s_{NN}} = 7.7$–$62.4$ GeV. At 7.7 GeV, $\langle\mu_B\rangle \approx 421$ MeV — closest to the hypothesized critical point location.
 
### Theoretical Pipeline
 
**Step 1 — Linear field mapping** (Parotto et al., Phys. Rev. C 101, 034901, 2020):
 
$$r = \frac{(T - T_c)\cos\alpha_2/T_c + (\mu_B - \mu_{Bc})\sin\alpha_2/T_c}{w\rho(\sin\alpha_1\cos\alpha_2 - \cos\alpha_1\sin\alpha_2)}$$
 
$$h = \frac{-(T - T_c)\cos\alpha_1/T_c - (\mu_B - \mu_{Bc})\sin\alpha_1/T_c}{-w(\sin\alpha_1\cos\alpha_2 - \cos\alpha_1\sin\alpha_2)}$$
 
Parameters: $T_c = 143.2$ MeV, $\mu_{Bc} = 350$ MeV, $\alpha_1 = 3.85°$, $\alpha_2 = 93.85°$, $w = 1.0$, $\rho = 2.0$.
 
**Step 2 — Correlation length** (Schofield parametrization, 3D Ising universality class):
 
$$\xi = \xi_0 \left(r^2 + |h|^{2/\delta}\right)^{-\nu/2}, \quad \nu = 0.630,\; \delta = 4.80,\; \xi_{\rm max} = 3 \text{ fm (non-equilibrium cap)}$$
 
**Step 3 — Cumulant enhancement** per freeze-out cell with embedding strength $s$:
 
$$\kappa_n^{\rm cell} = 1 + s\,\left(\xi^{\,\text{exponent}_n} - 1\right)$$
 
**Step 4 — Cooper-Frye particlization** with enhanced fluctuation weights via SMASH-hadron-sampler.
 
**Step 5 — Hadronic afterburner** via SMASH transport to produce final-state particles.
 
### BES Run Configuration
 
| Energy | $\langle\mu_B\rangle$ | Freeze-out Cells | Cells Near CP | Role |
|--------|----------------------|-----------------|---------------|------|
| 7.7 GeV | 421 MeV | 264,745 | 100,921 | Signal (label=1) |
| 11.5 GeV | 316 MeV | 362,521 | 132,960 | Control (label=1) |
| 19.6 GeV | 215 MeV | 670,229 | 15,801 | Baseline (label=0) |
 
---
 
## Results
 
### Classifier Performance (SMASH Afterburner Output)
 
| Metric | Value |
|--------|-------|
| Total jets | 8,055 |
| Signal jets (7.7 GeV) | 814 |
| Baseline jets (19.6 GeV) | 7,241 |
| Best AUC | 0.8692 |
 
### BES Energy Dependence
 
| Energy | $\mu_B$ (MeV) | Mean Classifier Score | Region |
|--------|---------------|----------------------|--------|
| 7.7 GeV | 421 | 0.72 | Above CP ($\mu_B > 350$ MeV) |
| 11.5 GeV | 316 | 0.50 | Below CP ($\mu_B < 350$ MeV) |
| 19.6 GeV | 215 | 0.28 | Far from CP |
 
The mean classifier score increases monotonically with $\mu_B$, consistent with the Ising embedding strength scaling with proximity to the critical point. The score crosses the random baseline (0.5) between 11.5 GeV ($\mu_B = 316$ MeV) and 7.7 GeV ($\mu_B = 421$ MeV), bracketing the conjectured critical point location at $\mu_{Bc} = 350$ MeV.

---
 
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
    │   ├── cluster_heavyion_jets.cc          # FastJet clustering on SMASH particle lists
    │   ├── cluster_heavyion_jets             # Compiled binary
    │   ├── train_heavyion.py                 # PyTorch: signal (7.7 GeV) vs baseline (19.6 GeV)
    │   ├── plot_afterburner_impact.py        # Sampler vs SMASH feature comparison
    │   ├── plot_BES_critical_point_signal.py # Score vs energy/muB
    │   │
    │   ├── jets_signal_7gev_smash.csv        # Clustered jets — 7.7 GeV (SMASH)
    │   ├── jets_control_11gev_smash.csv      # Clustered jets — 11.5 GeV (SMASH)
    │   ├── jets_baseline_19gev_smash.csv     # Clustered jets — 19.6 GeV (SMASH)
    │   ├── jets_signal_7gev.csv              # Clustered jets — 7.7 GeV (sampler)
    │   ├── jets_control_11gev.csv            # Clustered jets — 11.5 GeV (sampler)
    │   ├── jets_baseline_19gev.csv           # Clustered jets — 19.6 GeV (sampler)
    │   ├── jets_training.csv                 # Combined training set (signal + baseline)
    │   │
    │   ├── BES_critical_point_signal.png         # Score vs √s and μB — main physics result
    │   ├── afterburner_impact_analysis.png       # Sampler vs SMASH feature distributions
    │   ├── bes_energy_comparison.png             # Feature evolution across BES energies
    │   ├── heavyion_classifier_results.png       # ROC curves, loss, score distributions
    │   └── heavyion_classifier.pth               # Saved model weights
    │
    └── data/
        ├── signal_freezeout_7gev.dat     # Ising-embedded freeze-out, √s = 7.7  GeV
        ├── control_11gev.dat             # Ising-embedded freeze-out, √s = 11.5 GeV
        ├── baseline_19gev.dat            # Ising-embedded freeze-out, √s = 19.6 GeV
        │
        ├── vHlle-sampler BES II/         # Cooper-Frye sampled particle lists
        │   ├── signal_7gev/
        │   │   ├── particle_lists.oscar  # 1000 hadronic events, √s = 7.7  GeV
        │   │   └── 1000.root
        │   ├── control_11gev/
        │   │   ├── particle_lists.oscar  # 1000 hadronic events, √s = 11.5 GeV
        │   │   └── 1000.root
        │   └── baseline_19gev/
        │       ├── particle_lists.oscar  # 1000 hadronic events, √s = 19.6 GeV
        │       └── 1000.root
        │
        └── smash_output/                 # SMASH afterburner output
            ├── signal_7gev/
            │   ├── particle_lists.oscar
            │   └── config.yaml
            ├── control_11gev/
            │   └── particle_lists.oscar
            ├── baseline_19gev/
            │   └── particle_lists.oscar
            └── tabulations/
```
 
---
 
## Software Dependencies
 
| Package | Version | Purpose |
|---------|---------|---------|
| PYTHIA8 | 8.309+ | pp event generation |
| FastJet | ≥ 3.4 | Anti-$k_T$ jet clustering |
| PyTorch | ≥ 2.0 | Neural network training |
| vHLLE | latest | 3+1D viscous relativistic hydrodynamics |
| SMASH-hadron-sampler | v3.0 | Cooper-Frye particlization |
| SMASH | latest | Hadronic afterburner transport |
| NumPy / SciPy | latest | Numerical analysis |
| scikit-learn | latest | Preprocessing, ROC-AUC |
| pandas | latest | Data loading and management |
| matplotlib | latest | Plotting and visualization |
| CMake | ≥ 3.15 | C++ build system |
 
---
 
## License
 
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
 

