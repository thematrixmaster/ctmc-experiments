"""
Test if Full MLE can work at high epistasis with better optimization settings.

Goal: Keep unbounded epistasis but make Full MLE work through:
1. More training data
2. Better optimizer settings
3. Different learning rate schedules
"""

import sys
sys.path.append('/accounts/projects/yss/stephen.lu/ctmc')

from main import *

print("=" * 80)
print("Testing Full MLE at High Epistasis with Different Settings")
print("=" * 80)

epistasis = 12.0
seed = 42

gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)

print(f"\nGround truth: epistasis={epistasis}")
print(f"  Mean rate: {np.mean([gt.Q_global[i,j] for i in range(N_STATES) for j in range(N_STATES) if i!=j and hamming(SEQS[i], SEQS[j])==1]):.4f}")
print(f"  Rate ratio: {np.max([gt.Q_global[i,j] for i in range(N_STATES) for j in range(N_STATES) if i!=j and hamming(SEQS[i], SEQS[j])==1]) / np.min([gt.Q_global[i,j] for i in range(N_STATES) for j in range(N_STATES) if i!=j and hamming(SEQS[i], SEQS[j])==1]):.1f}×")

# Test different configurations
configs = [
    ("Baseline (10k samples, lr=0.05, wd=1e-4)", 10000, 0.05, 1e-4),
    ("More data (50k samples, lr=0.05, wd=1e-4)", 50000, 0.05, 1e-4),
    ("Even more data (100k samples, lr=0.05, wd=1e-4)", 100000, 0.05, 1e-4),
    ("Lower LR (100k samples, lr=0.01, wd=1e-4)", 100000, 0.01, 1e-4),
    ("Stronger reg (100k samples, lr=0.01, wd=1e-3)", 100000, 0.01, 1e-3),
    ("Very strong reg (100k samples, lr=0.01, wd=5e-3)", 100000, 0.01, 5e-3),
]

print("\n" + "=" * 80)
print("Testing Different Configurations")
print("=" * 80)

results = []

for name, n_samples, lr, wd in configs:
    print(f"\n{name}")
    print("-" * 80)

    # Generate data
    print(f"  Generating {n_samples} samples...")
    data = gt.generate_data(n_samples, lambda_param=0.5, min_mutations=0)

    # Train Full MLE
    print(f"  Training Full MLE (lr={lr}, wd={wd})...")
    model_full, history = train_full_mle_model(
        gt, data,
        epochs=500,
        lr=lr,
        verbose=False,
        early_stop_patience=50,
        early_stop_tolerance=0.001,
        weight_decay=wd
    )

    Q_full = model_full.build_Q()
    error_full = torch.norm(Q_full - Q_true) / torch.norm(Q_true)

    final_loss = history['losses'][-1]
    best_error = min(history['errors'])

    print(f"  ✓ Final error: {error_full.item():.4f}")
    print(f"  ✓ Best error during training: {best_error:.4f}")
    print(f"  ✓ Final loss: {final_loss:.4f}")
    print(f"  ✓ Epochs trained: {len(history['losses'])}")

    results.append({
        'name': name,
        'n_samples': n_samples,
        'lr': lr,
        'wd': wd,
        'error': error_full.item(),
        'best_error': best_error,
        'final_loss': final_loss,
        'epochs': len(history['losses'])
    })

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Configuration':<50} {'Final Error':<15} {'Best Error':<15} {'Epochs':<10}")
print("-" * 95)
for r in results:
    print(f"{r['name']:<50} {r['error']:<15.4f} {r['best_error']:<15.4f} {r['epochs']:<10}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Find best configuration
best_config = min(results, key=lambda x: x['error'])
print(f"\nBest configuration: {best_config['name']}")
print(f"  Error: {best_config['error']:.4f}")

# Check if more data helps
data_10k = [r for r in results if r['n_samples'] == 10000][0]
data_100k = [r for r in results if r['n_samples'] == 100000 and r['lr'] == 0.05][0]

improvement = (data_10k['error'] - data_100k['error']) / data_10k['error'] * 100
print(f"\nEffect of 10× more data (10k → 100k):")
print(f"  Error reduction: {improvement:.1f}%")
print(f"  From {data_10k['error']:.4f} → {data_100k['error']:.4f}")

if improvement > 30:
    print(f"  → More data HELPS significantly! Consider even more data.")
elif improvement > 10:
    print(f"  → More data helps moderately.")
else:
    print(f"  → More data doesn't help much. Problem is optimization, not data.")

# Check if regularization helps
reg_weak = [r for r in results if r['wd'] == 1e-4 and r['n_samples'] == 100000 and r['lr'] == 0.01][0]
reg_strong = [r for r in results if r['wd'] == 1e-3 and r['n_samples'] == 100000][0]

reg_improvement = (reg_weak['error'] - reg_strong['error']) / reg_weak['error'] * 100
print(f"\nEffect of 10× stronger regularization (1e-4 → 1e-3):")
print(f"  Error reduction: {reg_improvement:.1f}%")
print(f"  From {reg_weak['error']:.4f} → {reg_strong['error']:.4f}")

if reg_improvement > 20:
    print(f"  → Regularization HELPS significantly!")
elif reg_improvement > 5:
    print(f"  → Regularization helps moderately.")
else:
    print(f"  → Regularization doesn't help much.")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

if best_config['error'] < 0.10:
    print(f"\n✓ Full MLE CAN work at epistasis={epistasis}!")
    print(f"  Best error: {best_config['error']:.4f}")
    print(f"  Use: n_samples={best_config['n_samples']}, lr={best_config['lr']}, wd={best_config['wd']}")
else:
    print(f"\n✗ Full MLE struggles even with best settings")
    print(f"  Best error: {best_config['error']:.4f}")
    print(f"\nPossible solutions:")
    print(f"  1. Use even more data (200k+ samples)")
    print(f"  2. Use a better optimizer (L-BFGS, second-order methods)")
    print(f"  3. Use curriculum learning (start with low epistasis, gradually increase)")
    print(f"  4. Modify the loss function (add penalties for extreme rates)")

print("\nTest complete!")
