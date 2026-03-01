# Quark–Gluon Jet Binary Classification Project

## Motivation
As the Quark- and gluon-initiated jets exhibit subtle differences due to QCD color factors:

- Gluons radiate more strongly (CA = 3)
- Quarks radiate less (CF = 4/3)

These differences lead to measurable variations in jet substructure observables such as:

- Particle multiplicity (charged and neutral particles)
- Jet mass
- N-subjettiness ratios (τ21)
- Jet width (Girth)
- jet eccentricity

This project builds a Monte Carlo simulation for p-p collision at 13 TeV using **PYTHIA8** event generation , then Jet clustering using **FastJet** set-up to extract certain jet features that trained by the neural network and Binary cross entropy classifier using **PyTorch** toolkit. The results had evaluated using the ROC/AUC ratio curve.






