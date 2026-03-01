"""
Binary Classifier for the quark/gluon jets due to the Jet structure features
"""

import torch
import torch.nn as nn
import torch.optim as optim
#sklearn modules for data splitting, scaling, and evaluation metrics
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        if desc:
            print(desc)
        return iterable

# Check for display environment
import matplotlib
if 'DISPLAY' not in os.environ:
    print("No display detected. Using 'Agg' backend (plots saved only).")
    matplotlib.use('Agg')

print("\nLoading PYTHIA/FastJet jet dataset...")

data = np.loadtxt("jets_pythia.csv", delimiter=",", skiprows=1)

X = data[:, :-1]
y = data[:, -1]

# Log-transform skewed variables
X[:,0] = np.log(X[:,0] + 1)  # mass
X[:,1] = np.log(X[:,1] + 1)  # n_charged
X[:,2] = np.log(X[:,2] + 1)  # n_neutral
X[:,5] = np.log(X[:,5] + 1)  # girth
X[:,6] = np.log(X[:,6] + 1)  # eccentricity
print("Quarks:", sum(y==0))
print("Gluons:", sum(y==1))

# feature names
feature_names = [
    'Mass',
    'n_charged',     # charged multiplicity
    'n_neutral',     # neutral multiplicity
    'tau21',
    'tau32',
    'Girth',
    'eccentricity'   # jet shape
]

#Data Visualization
def plot_physics_comparison(X, y, feature_names):
    """Plot feature distributions """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    quark_mask = y.flatten() == 0
    gluon_mask = y.flatten() == 1
     #for each observable
    for i in range(min(5, X.shape[1])):
        axes[i].hist(X[quark_mask, i], bins=20, alpha=0.7,
                     label=f'Quarks (n={sum(quark_mask)})',
                     color='blue', density=True)
        axes[i].hist(X[gluon_mask, i], bins=20, alpha=0.7,
                     label=f'Gluons (n={sum(gluon_mask)})',
                     color='red', density=True)
        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel('Normalized Frequency')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
                     
    # Mass vs Girth scatter plot for radiation sensitivity
    axes[5].scatter(X[quark_mask, 5], X[quark_mask, 0],
                alpha=0.3, c='blue', s=10, label='Quarks')
    axes[5].scatter(X[gluon_mask, 5], X[gluon_mask, 0],
                alpha=0.3, c='red', s=10, label='Gluons')
    axes[5].set_xlabel('Girth')
    axes[5].set_ylabel('Mass (GeV)')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
  
    plt.tight_layout()
    plt.savefig('observable_features.png', dpi=100)
    plt.show()

def plot_training_results(train_losses, val_losses, y_test, preds, X_test, scaler):
    """Plot loss curves, ROC, prediction distribution, and jet features"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Loss curves
    axes[0,0].plot(train_losses, label='Training', linewidth=2)
    axes[0,0].plot(val_losses, label='Validation', linewidth=2)
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Progress')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0,1].plot([0,1], [0,1], 'k--', label='Random')
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

# Estimating threshold using Youden's J statistic
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal decision threshold: {optimal_threshold:.3f}")

    # Physics interpretation text
    if roc_auc > 0.8:
        perf_text = "EXCELLENT discrimination"
    elif roc_auc > 0.7:
        perf_text = "GOOD discrimination"
    elif roc_auc > 0.6:
        perf_text = "MODERATE discrimination"
    else:
        perf_text = "POOR discrimination\nFeatures need improvement"
    axes[0,1].text(0.6, 0.2, perf_text,
                   transform=axes[0,1].transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 3. Prediction distribution
    axes[1,0].hist(preds[y_test.flatten()==0], bins=20, alpha=0.7,
                   label='True Quarks', color='blue', density=True)
    axes[1,0].hist(preds[y_test.flatten()==1], bins=20, alpha=0.7,
                   label='True Gluons', color='red', density=True)
    axes[1,0].axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Decision Threshold: {optimal_threshold:.2f}')

    axes[1,0].set_xlabel('Predicted Probability')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. jet Features correlated with prediction 
    x_pos = np.arange(X_test.shape[1])
    feature_names_short = feature_names
    corr_all = []
    # Rescale X_test back to original scale
    X_test_orig = X_test * scaler.scale_ + scaler.mean_

    for i in range(X_test_orig.shape[1]):
        corr = np.corrcoef(X_test_orig[:, i], preds.flatten())[0, 1]
        corr_all.append(corr)
    axes[1,1].bar(x_pos, corr_all, width=0.6, color='purple', alpha=0.7)
    axes[1,1].set_xlabel('jet Features')
    axes[1,1].set_ylabel('Correlation with Prediction')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(feature_names_short, rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=100)
    plt.show()
    return roc_auc

def main():
    print("="*60)
    print("JET SUBSTRUCTURE CLASSIFIER")
    print("="*60)
    print(f"Python version: torch.__version__ = {torch.__version__}")
    # Memory‑based configuration selection
    n_jets = 1000
    epochs = 35
        
    # Normalise to standard scale  (zero mean and unit variance )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Physics feature plots ---
    print("\n" + "-"*60)
    print("Displaying physics feature plots.")
    plot_physics_comparison(X, y, feature_names)


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\nTraining set: {len(X_train)} jets")
    print(f"Validation set: {len(X_val)} jets")
    print(f"Test set: {len(X_test)} jets")

    # datasets convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)

    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)

    X_test = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

       # MODEL naural network
    class JetClassifier(nn.Module):
        def __init__(self):
            super(JetClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(X_train.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(64, 32),
                nn.ReLU(),

                nn.Linear(32, 16),
                nn.ReLU(),

                nn.Linear(16, 1),
                #nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)


    model = JetClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #to balance datasets weights
    pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  
    # Training loop
    train_losses = []
    val_losses = []
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)
            val_losses.append(val_loss.item())

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f} - Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")

    # Evaluation
    print("\nEvaluating model...")
    model.eval()

    with torch.no_grad():
       logits = model(X_test)
       preds = torch.sigmoid(logits).numpy()

    # --- Training results plot ---
    print("\n" + "-"*60)
    print("Displaying training results plots.")
    print("If they don't appear, check the backend. Files are also saved.")
    print("-"*60)
    auc_score = plot_training_results(
    train_losses,
    val_losses,
    y_test,
    preds,
    X_test, 
    scaler           
      )
    # --- AUC validation ---
    if auc_score < 0.6:
        print("\n WARNING poor separation.")
        print("   - Data generation parameters may need tuning")
        print("   - Consider adding more discriminating observables.")
    else:
        print(f"\n Good separation: AUC = {auc_score:.3f}")

    print("\n" + "="*60)
    print(f"AUC Score: {auc_score:.3f}")

    if auc_score > 0.8:
        print(" The model successfully distinguishes quark and gluon jets")
    elif auc_score > 0.7:
        print(" Good discrimination - physics features are effective")
        print(" Some overlap in jet properties is expected from QCD")
    else:
        print(" Limited discrimination - consider additional observables")
      
    # Accuracy of Binary classifier 
    preds_binary = (preds > 0.5).astype(int)
    quark_acc = np.mean(preds_binary[y_test.flatten()==0] == 0) if sum(y_test.flatten()==0) > 0 else 0
    gluon_acc = np.mean(preds_binary[y_test.flatten()==1] == 1) if sum(y_test.flatten()==1) > 0 else 0
    print(f"\nClassification Accuracy:")
    print(f"  Quarks: {quark_acc:.1%}")
    print(f"  Gluons: {gluon_acc:.1%}")

    # Save model
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': scaler.mean_,
            'scaler_scale': scaler.scale_,
            'auc_score': auc_score
        }, 'physics_jet_classifier.pth')
        print("\n Model saved as 'physics_jet_classifier.pth'")
    except Exception as e:
        print(f"\n Could not save model: {e}")

   
#Main exceution
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
    except Exception as e:
        print(f"\nError occurred: {e}")
