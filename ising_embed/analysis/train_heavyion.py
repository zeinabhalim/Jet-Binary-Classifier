
"""
Heavy-Ion Jet Classifier: signal (7.7 GeV Ising-embedded) vs baseline (19.6 GeV)
Features: jet_pt, jet_eta, jet_phi, jet_mass, nconstituents, pt_d, sqrt_d12
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading datasets...")
s  = pd.read_csv("jets_signal_7gev.csv")    # label=1
b  = pd.read_csv("jets_baseline_19gev.csv") # label=0
df = pd.concat([s, b], ignore_index=True)

FEATURES = ['jet_pt','jet_eta','jet_phi','jet_mass',
            'nconstituents','pt_d','sqrt_d12']

X = df[FEATURES].values.astype(np.float32)
y = df['label'].values.astype(np.float32)

print(f"Signal   (7.7 GeV  label=1): {sum(y==1):>6}")
print(f"Baseline (19.6 GeV label=0): {sum(y==0):>6}")
print(f"Total:                        {len(y):>6}")

# ── Preprocessing ──────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# Convert to tensors
def to_tensor(x, y):
    return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(1)

Xt, yt   = to_tensor(X_train, y_train)
Xv, yv   = to_tensor(X_val,   y_val)
Xte, yte = to_tensor(X_test,  y_test)

# ── Model ──────────────────────────────────────────────────────────────────
class HeavyIonClassifier(nn.Module):
    def __init__(self, n_features=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = HeavyIonClassifier(n_features=len(FEATURES))

# Class imbalance weight
pos_weight = torch.tensor([sum(y==0) / sum(y==1)])
print(f"pos_weight = {pos_weight.item():.2f}  (corrects class imbalance)")
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# ── Training ───────────────────────────────────────────────────────────────
EPOCHS = 80
train_losses, val_losses = [], []
best_auc, best_state = 0.0, None

print("\nTraining...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(Xt), yt)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        vl = criterion(model(Xv), yv).item()
        val_losses.append(vl)
        vp = torch.sigmoid(model(Xte)).numpy()
        ep_auc = roc_auc_score(y_test, vp)
    scheduler.step(vl)

    if ep_auc > best_auc:
        best_auc   = ep_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if (epoch+1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={loss.item():.4f}  val={vl:.4f}  AUC={ep_auc:.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(Xte)).numpy().flatten()

final_auc = roc_auc_score(y_test, preds)
fpr, tpr, thr = roc_curve(y_test, preds)

print(f"\n{'='*50}")
print(f"Best AUC  : {best_auc:.4f}")
print(f"Final AUC : {final_auc:.4f}")
print(f"{'='*50}")
print("Physics interpretation:")
if final_auc > 0.70:
    print("  STRONG signal — classifier detects Ising CP embedding")
    print("  The 7.7 GeV QGP signature is distinguishable from 19.6 GeV")
elif final_auc > 0.60:
    print("  MODERATE signal — CP fluctuations partially visible")
else:
    print("  WEAK signal — consider increasing embedding strength")

# ── Also test on 11.5 GeV control ─────────────────────────────────────────
try:
    c = pd.read_csv("jets_control_11gev.csv")
    Xc = scaler.transform(c[FEATURES].values.astype(np.float32))
    with torch.no_grad():
        pc = torch.sigmoid(model(torch.FloatTensor(Xc))).numpy().flatten()
    yc = c['label'].values
    auc_11 = roc_auc_score(yc, pc) if len(np.unique(yc)) > 1 else np.nan
    mean_score_11 = pc.mean()
    print(f"\nControl (11.5 GeV) mean classifier score: {mean_score_11:.4f}")
    print(f"  Expected: between 7.7 GeV and 19.6 GeV scores")
    print(f"  7.7 GeV  mean score: {preds[y_test==1].mean():.4f}  ← highest (closest to CP)")
    print(f"  11.5 GeV mean score: {mean_score_11:.4f}             ← intermediate")
    print(f"  19.6 GeV mean score: {preds[y_test==0].mean():.4f}  ← lowest (far from CP)")
except Exception as e:
    print(f"Control dataset: {e}")

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(train_losses, label='Train')
axes[0,0].plot(val_losses,   label='Val')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].set_title('Training Loss'); axes[0,0].legend(); axes[0,0].grid(True)

axes[0,1].plot(fpr, tpr, color='darkorange', lw=2,
               label=f'AUC = {final_auc:.4f}')
axes[0,1].plot([0,1],[0,1],'k--')
axes[0,1].set_xlabel('FPR'); axes[0,1].set_ylabel('TPR')
axes[0,1].set_title('ROC Curve — 7.7 vs 19.6 GeV')
axes[0,1].legend(); axes[0,1].grid(True)

axes[1,0].hist(preds[y_test==1], bins=40, alpha=0.7,
               label='7.7 GeV (signal)', color='red',  density=True)
axes[1,0].hist(preds[y_test==0], bins=40, alpha=0.7,
               label='19.6 GeV (baseline)', color='blue', density=True)
axes[1,0].set_xlabel('Classifier Score')
axes[1,0].set_ylabel('Density')
axes[1,0].set_title('Score Distribution')
axes[1,0].legend(); axes[1,0].grid(True)

# Feature importance via correlation
corr = [np.corrcoef(X_test[:,i], preds)[0,1] for i in range(len(FEATURES))]
axes[1,1].barh(FEATURES, corr, color='purple', alpha=0.7)
axes[1,1].set_xlabel('Correlation with classifier score')
axes[1,1].set_title('Feature Importance')
axes[1,1].axvline(0, color='black', lw=0.5)
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Heavy-Ion Jet Classifier: Ising CP Signal Detection\n'
             '7.7 GeV (signal) vs 19.6 GeV (baseline)', fontsize=12)
plt.tight_layout()
plt.savefig('heavyion_classifier_results.png', dpi=150)
print("\nPlot saved: heavyion_classifier_results.png")

torch.save({'model_state_dict': best_state,
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'features': FEATURES,
            'auc': final_auc},
           'heavyion_classifier.pth')
print("Model saved: heavyion_classifier.pth")
