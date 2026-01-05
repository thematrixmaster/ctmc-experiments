"""
Test the proposed fixes for full MLE underperformance.

Tests:
1. Increased data (30k samples)
2. Reduced learning rate (0.005)
3. Increased early stopping patience (100)
4. Added L2 regularization

Expected: Full MLE should match or beat factorized at epistasis=0.0
"""

from main import *
import torch.nn.functional as F

def train_full_mle_model_fixed(gt, train_data, epochs=500, lr=0.005, verbose=True, 
                               bucket_decimals=2, early_stop_patience=100, 
                               early_stop_tolerance=0.005, weight_decay=1e-4):
    """
    Fixed version of train_full_mle_model with:
    - Lower learning rate (0.005 instead of 0.01)
    - More patience (100 instead of 50)
    - More lenient tolerance (0.005 instead of 0.001)
    - L2 regularization (weight_decay=1e-4)
    """
    import time
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    model = FullMLEModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Added weight_decay

    # Pre-process data and move to device
    starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in train_data], dtype=torch.long).to(DEVICE)
    ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in train_data], dtype=torch.long).to(DEVICE)
    bs = torch.tensor([d[2] for d in train_data], dtype=torch.float32).to(DEVICE)

    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
    history = {'losses': [], 'errors': [], 'epoch_times': []}

    # Early stopping setup
    best_error = float('inf')
    patience_counter = 0
    best_model_state = None

    # Bucket setup
    bucket_factor = 10 ** bucket_decimals
    bs_bucketed = torch.round(bs * bucket_factor) / bucket_factor
    unique_bs = torch.unique(bs_bucketed)
    n_buckets = len(unique_bs)

    if verbose:
        print(f"Training Full MLE Model (FIXED):")
        print(f"  Samples: {len(train_data)}")
        print(f"  Max epochs: {epochs}")
        print(f"  Early stopping: patience={early_stop_patience}, tolerance={early_stop_tolerance}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay (L2): {weight_decay}")
        print(f"  Unique branch lengths: {n_buckets}")

    epoch_iterator = tqdm(range(epochs), desc="Training") if has_tqdm else range(epochs)

    for epoch in epoch_iterator:
        epoch_start = time.time()
        optimizer.zero_grad()

        # Build current Q matrix
        Q = model.build_Q()

        # Compute likelihood for each observation
        log_prob_total = 0

        for b_val in unique_bs:
            mask = (bs_bucketed == b_val)
            if mask.sum() == 0:
                continue

            # Matrix exponential
            P = torch.matrix_exp(b_val * Q)

            # Get probabilities for this batch
            batch_starts = starts_idx[mask]
            batch_ends = ends_idx[mask]

            probs = P[batch_starts, batch_ends]
            log_prob_total += torch.sum(torch.log(probs + 1e-9))

        loss = -log_prob_total / len(train_data)
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        history['epoch_times'].append(epoch_time)

        # Track metrics every epoch for early stopping
        with torch.no_grad():
            Q_model = model.build_Q()
            error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

            history['losses'].append(loss.item())
            history['errors'].append(error.item())

            # Early stopping check
            if error < best_error - early_stop_tolerance:
                best_error = error.item()
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            # Display progress
            if verbose and not has_tqdm and epoch % 10 == 0:
                print(f"  Epoch {epoch:4d}/{epochs}: Loss={loss.item():.4f}, Error={error.item():.4f} "
                      f"Patience={patience_counter}/{early_stop_patience}")

            if has_tqdm and epoch % 5 == 0:
                epoch_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'error': f'{error.item():.4f}',
                    'patience': f'{patience_counter}/{early_stop_patience}',
                })

            # Check early stopping
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch} (best error: {best_error:.4f})")
                model.load_state_dict(best_model_state)
                break

    return model, history

def test_with_different_data_sizes():
    """Test full MLE vs factorized with different data sizes."""
    print("="*70)
    print("TEST: Full MLE vs Factorized with Different Data Sizes")
    print("="*70)
    
    epistasis = 0.0  # Test at epistasis=0 where both should be similar
    seed = 42
    sample_sizes = [10000, 20000, 30000]
    
    results = []
    
    for n_samples in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with {n_samples} samples")
        print(f"{'='*70}")
        
        # Generate data
        gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
        Q_true = torch.tensor(gt.Q_global, dtype=torch.float32).to(DEVICE)
        
        print(f"Generating {n_samples} samples...")
        data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=0)
        print(f"Generated {len(data)} samples")
        
        # Train factorized model
        print(f"\n1. Training factorized model...")
        model_fact, history_fact = train_model(
            gt, data, 
            use_snr=False, 
            epochs=1000, 
            lr=0.01,
            early_stop_patience=100, 
            early_stop_tolerance=0.001, 
            verbose=False
        )
        Q_fact = model_fact.build_global_Q()
        error_fact = torch.norm(Q_fact - Q_true) / torch.norm(Q_true)
        epochs_fact = len(history_fact['epochs'])
        print(f"   Error: {error_fact.item():.4f}")
        print(f"   Epochs: {epochs_fact}")
        
        # Train full MLE (ORIGINAL)
        print(f"\n2. Training full MLE (ORIGINAL parameters)...")
        model_full_orig, history_orig = train_full_mle_model(
            gt, data,
            epochs=500,
            lr=0.01,
            verbose=False,
            early_stop_patience=50,
            early_stop_tolerance=0.001
        )
        Q_full_orig = model_full_orig.build_Q()
        error_full_orig = torch.norm(Q_full_orig - Q_true) / torch.norm(Q_true)
        epochs_orig = len(history_orig['errors'])
        print(f"   Error: {error_full_orig.item():.4f}")
        print(f"   Epochs: {epochs_orig}")
        
        # Train full MLE (FIXED)
        print(f"\n3. Training full MLE (FIXED parameters)...")
        model_full_fixed, history_fixed = train_full_mle_model_fixed(
            gt, data,
            epochs=500,
            lr=0.005,  # Reduced
            verbose=False,
            early_stop_patience=100,  # Increased
            early_stop_tolerance=0.005,  # More lenient
            weight_decay=1e-4  # Added regularization
        )
        Q_full_fixed = model_full_fixed.build_Q()
        error_full_fixed = torch.norm(Q_full_fixed - Q_true) / torch.norm(Q_true)
        epochs_fixed = len(history_fixed['errors'])
        print(f"   Error: {error_full_fixed.item():.4f}")
        print(f"   Epochs: {epochs_fixed}")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"Summary for {n_samples} samples:")
        print(f"{'='*70}")
        print(f"  Factorized:          {error_fact.item():.4f} ({epochs_fact} epochs)")
        print(f"  Full MLE (original): {error_full_orig.item():.4f} ({epochs_orig} epochs)")
        print(f"  Full MLE (fixed):    {error_full_fixed.item():.4f} ({epochs_fixed} epochs)")
        print(f"\n  Improvement: {error_full_orig.item() - error_full_fixed.item():+.4f}")
        
        if error_full_fixed.item() < error_fact.item():
            print(f"  ✓ Full MLE (fixed) beats factorized!")
        elif error_full_fixed.item() < error_fact.item() + 0.02:
            print(f"  ✓ Full MLE (fixed) matches factorized")
        else:
            print(f"  ⚠ Full MLE (fixed) still underperforms")
        
        results.append({
            'n_samples': n_samples,
            'error_fact': error_fact.item(),
            'error_full_orig': error_full_orig.item(),
            'error_full_fixed': error_full_fixed.item(),
            'epochs_fact': epochs_fact,
            'epochs_orig': epochs_orig,
            'epochs_fixed': epochs_fixed
        })
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Samples':<10} {'Factorized':<12} {'Full (orig)':<12} {'Full (fixed)':<12} {'Status'}")
    print("-"*70)
    
    for r in results:
        if r['error_full_fixed'] < r['error_fact'] + 0.02:
            status = "✓ Fixed"
        else:
            status = "⚠ Still worse"
        print(f"{r['n_samples']:<10} {r['error_fact']:<12.4f} {r['error_full_orig']:<12.4f} "
              f"{r['error_full_fixed']:<12.4f} {status}")
    
    # Plot comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sample_sizes = [r['n_samples'] for r in results]
    
    # Plot 1: Errors
    ax = axes[0]
    ax.plot(sample_sizes, [r['error_fact'] for r in results], 
            'o-', linewidth=2, markersize=8, label='Factorized')
    ax.plot(sample_sizes, [r['error_full_orig'] for r in results], 
            's-', linewidth=2, markersize=8, label='Full MLE (original)')
    ax.plot(sample_sizes, [r['error_full_fixed'] for r in results], 
            'd-', linewidth=2, markersize=8, label='Full MLE (fixed)')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Relative Frobenius Error', fontsize=12)
    ax.set_title('Error vs Sample Size (epistasis=0.0)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training epochs
    ax = axes[1]
    ax.plot(sample_sizes, [r['epochs_fact'] for r in results], 
            'o-', linewidth=2, markersize=8, label='Factorized')
    ax.plot(sample_sizes, [r['epochs_orig'] for r in results], 
            's-', linewidth=2, markersize=8, label='Full MLE (original)')
    ax.plot(sample_sizes, [r['epochs_fixed'] for r in results], 
            'd-', linewidth=2, markersize=8, label='Full MLE (fixed)')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Epochs Until Convergence', fontsize=12)
    ax.set_title('Training Efficiency', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/fix_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results/fix_comparison.png")

if __name__ == "__main__":
    test_with_different_data_sizes()


