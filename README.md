# Quark–Gluon Jet Binary Classification Project

 Quark- and gluon-initiated jets exhibit subtle differences arising from Quantum Chromodynamics (QCD) color factors. As the gluons carry a larger color charge ( \( C_A = 3 \) ) than quarks ( \( C_F = 4/3 \) ) therefore gluon jets radiate more intensely. This enhanced radiation leads to broader jets with higher particle multiplicity and variations in jet substructure observables such as:

- Particle multiplicity (charged and neutral constituents)
- Jet mass
- N-subjettiness ratios (τ21)
- Jet width (Girth)
- jet eccentricity

Accordingly, the combined correlations among these observables allow statistical discrimination between quark and gluon jets that implemented in this projrct as follows;

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

---

## Repository Structure


JetBinaryClass/
│
├── CMakeLists.txt # C++ build configuration for PYTHIA8 + FastJet
├── LICENSE 
├── README.md 
│
├── src/
│ └── generate_pythia_jets.cc
│ # Event generation and jet feature extraction
│
├── MLmodel/
│ └── train.py # Neural network training and evaluation

---

## Build Instructions
1. 
```bash
mkdir build
cd build
cmake ..
make

2. Run the event generator:

./generate_pythia_jets

**This produces the dataset file (e.g., jets_pythia.npz).**

3. Run your Training machine learning model:

cd MLmodel
python train.py

Outputs include: observable_features.png, training_results.png , and physics_jet_classifier.pth


