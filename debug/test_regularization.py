"""
Test L2 regularization with 30k samples.

Compare:
1. Full MLE without regularization (baseline from previous test: 0.1154)
2. Full MLE with different L2 regularization strengths

Goal: See if regularization can help full MLE match factorized (~0.075)
"""

from main import *
import time

def train_full_mle_with_regularization(gt, train_data, weight_decay=0.0, epochs=500, 
                                       lr=0.01, verbose=True):
    """Train full MLE with L2 regularization."""
    import time
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    model = FullMLEModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Pre-process data
    starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in train_data], 
                              dtype=torch.long).to(DEVICE)
    ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in train_data], 
                            dtype=torch.long).to(DEVICE)
    bs = torch.tensor([d[2] for d in train_data], dtype=torch.float32).to(DEVICE)

    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    history = {'losses': [], 'errors': [], 'epoch_times': []}

    # Early stopping setup
    best_error = float('inf')
    patience_counter = 0
    best_model_state = None
    early_stop_patience = 50
    early_stop_tolerance = 0.001

    # Bucket setup
    bucket_decimals = 2
    bucket_factor = 10 ** bucket_decimals
    bs_bucketed = torch.round(bs * bucket_factor) / bucket_factor
    unique_bs = torch.unique(bs_bucketed)

    if verbose:
        print(f"Training Full MLE Model:")
        print(f"  Samples: {len(train_data)}")
        print(f"  Max epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay (L2): {weight_decay}")
        print(f"  Early stopping: patience={early_stop_patience}, tolerance={early_stop_tolerance}")

    epoch_iterator = tqdm(range(epochs), desc="Training") if has_tqdm else range(epochs)

    for epoch in epoch_iterator:
        epoch_start = time.time()
        optimizer.zero_grad()

        Q = model.build_Q()
        log_prob_total = 0

        for b_val in unique_bs:
            mask = (bs_bucketed == b_val)
            if mask.sum() == 0:
                continue
            P = torch.matrix_exp(b_val * Q)
            batch_starts = starts_idx[mask]
            batch_ends = ends_idx[mask]
            probs = P[batch_starts, batch_ends]
            log_prob_total += torch.sum(torch.log(probs + 1e-9))

        loss = -log_prob_total / len(train_data)
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        with torch.no_grad():
            Q_model = model.build_Q()
            error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)
            history['losses'].append(loss.item())
            history['errors'].append(error.item())

            if error < best_error - early_stop_tolerance:
                best_error = error.item()
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if has_tqdm and epoch % 5 == 0:
                epoch_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'error': f'{error.item():.4f}',
                    'patience': f'{patience_counter}/{early_stop_patience}',
                })

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch} (best error: {best_error:.4f})")
                model.load_state_dict(best_model_state)
                break

    return model, history

if __name__ == "__main__":
    print("="*70)
    print("TEST: L2 Regularization with 30k Samples")
    print("="*70)
    print("\nGoal: See if regularization helps full MLE match factorized model")
    print("Factorized model error (reference): 0.0750")
    print("Full MLE without regularization: 0.1154")
    
    # Setup
    epistasis = 0.0
    seed = 42
    epochs = 500
    n_samples = 50000
    lr = 0.005
    
    print(f"\n{'='*70}")
    print("Setup")
    print(f"{'='*70}")
    print(f"  Epistasis: {epistasis}")
    print(f"  Seed: {seed}")
    print(f"  Samples: {n_samples}")
    
    # Generate data
    print(f"\nGenerating ground truth and data...")
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    
    start_time = time.time()
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)
    gen_time = time.time() - start_time
    print(f"Generated {len(data)} samples in {gen_time:.1f}s")
    
    # Test different regularization strengths
    weight_decays = [0.0001]  # Just baseline vs recommended L2
    results = []
    
    FACTORIZED_ERROR = 0.075
    
    for wd in weight_decays:
        print(f"\n{'='*70}")
        print(f"Testing weight_decay = {wd}")
        print(f"{'='*70}")
        
        model, history = train_full_mle_with_regularization(
            gt, data,
            weight_decay=wd,
            epochs=epochs,
            lr=lr,
            verbose=True
        )
        
        with torch.no_grad():
            Q_model = model.build_Q()
            final_error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)
        
        best_error = min(history['errors']) if history['errors'] else final_error.item()
        epochs_trained = len(history['errors'])
        train_time = sum(history['epoch_times']) / 60  # in minutes
        
        print(f"\n  Results:")
        print(f"    Best error: {best_error:.4f}")
        print(f"    Final error: {final_error.item():.4f}")
        print(f"    Epochs: {epochs_trained}")
        print(f"    Training time: {train_time:.1f} min")
        print(f"    Gap to factorized: {best_error - FACTORIZED_ERROR:+.4f}")
        
        results.append({
            'weight_decay': wd,
            'best_error': best_error,
            'final_error': final_error.item(),
            'epochs': epochs_trained,
            'train_time': train_time,
            'gap': best_error - FACTORIZED_ERROR
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Effect of L2 Regularization")
    print(f"{'='*70}")
    print(f"\n{'Weight Decay':<15} {'Best Error':<12} {'Gap to Fact':<12} {'Epochs':<10} {'Status'}")
    print("-"*70)
    
    for r in results:
        if r['gap'] < 0.02:
            status = "✓ Matches!"
        elif r['gap'] < 0.04:
            status = "~ Close"
        else:
            status = "✗ Still worse"
        
        wd_str = f"{r['weight_decay']:.0e}" if r['weight_decay'] > 0 else "0"
        print(f"{wd_str:<15} {r['best_error']:<12.4f} {r['gap']:>+10.4f}  {r['epochs']:<10} {status}")
    
    # Find best
    best_result = min(results, key=lambda x: x['best_error'])
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    print(f"\nBest configuration:")
    print(f"  Weight decay: {best_result['weight_decay']}")
    print(f"  Error: {best_result['best_error']:.4f}")
    print(f"  Gap to factorized: {best_result['gap']:+.4f}")
    
    baseline = results[0]  # no regularization
    improvement = baseline['best_error'] - best_result['best_error']
    
    if improvement > 0.001:
        print(f"\nRegularization helped!")
        print(f"  Improvement: {improvement:.4f} ({improvement/baseline['best_error']*100:.1f}% reduction)")
    else:
        print(f"\nRegularization didn't help much.")
        print(f"  Change: {improvement:+.4f}")
    
    if best_result['gap'] < 0.02:
        print(f"\n✓✓✓ SUCCESS! ✓✓✓")
        print(f"\nWith 30k samples and L2 regularization (weight_decay={best_result['weight_decay']}),")
        print(f"full MLE achieves {best_result['best_error']:.4f}, matching factorized ({FACTORIZED_ERROR:.4f})!")
        print(f"\nRecommendation: Update main.py to use weight_decay={best_result['weight_decay']}")
        
    elif improvement > 0.02:
        print(f"\n⚠ PARTIAL SUCCESS")
        print(f"\nRegularization helped reduce error by {improvement:.4f},")
        print(f"but full MLE still underperforms factorized by {best_result['gap']:.4f}.")
        print(f"\nNext steps:")
        print(f"  1. Try more data (50k samples)")
        print(f"  2. Try lower learning rate (0.005)")
        print(f"  3. Combine regularization with more data")
        
    else:
        print(f"\n✗ NO IMPROVEMENT")
        print(f"\nL2 regularization didn't help with current sample size.")
        print(f"The issue is likely insufficient data coverage.")
        print(f"\nRecommendation: Increase to 50k+ samples")
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Error vs weight decay
    ax = axes[0]
    wd_values = [r['weight_decay'] for r in results]
    errors = [r['best_error'] for r in results]
    
    # Use log scale for x-axis (skip 0)
    wd_plot = [i for i in range(len(wd_values))]
    labels = ['0'] + [f'{wd:.0e}' for wd in wd_values[1:]]
    
    ax.plot(wd_plot, errors, 'o-', linewidth=2, markersize=8, color='blue')
    ax.axhline(y=FACTORIZED_ERROR, color='orange', linestyle='--', 
               linewidth=2, label=f'Factorized ({FACTORIZED_ERROR:.4f})')
    ax.set_xticks(wd_plot)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('L2 Weight Decay', fontsize=12)
    ax.set_ylabel('Best Error', fontsize=12)
    ax.set_title('Full MLE: Effect of L2 Regularization', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training curves for best regularization
    ax = axes[1]
    best_idx = results.index(best_result)
    
    # Re-plot training history (would need to save it, for now just show final)
    ax.text(0.5, 0.5, f'Best Config:\nweight_decay={best_result["weight_decay"]}\n'
            f'Error: {best_result["best_error"]:.4f}\n'
            f'Gap: {best_result["gap"]:+.4f}',
            transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('Best Configuration', fontsize=13)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/regularization_test.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results/regularization_test.png")

