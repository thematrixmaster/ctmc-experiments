"""
Debug script to check if unobserved transitions are causing the problem.
"""
import sys
sys.path.insert(0, '/Users/stephen/Desktop/ctmc')
from main import *

print("=" * 70)
print("DEBUGGING: Transition Coverage vs Error")
print("=" * 70)

# Test with different epistasis levels
for eps in [0.0, 1.0]:
    print(f"\nEpistasis = {eps}")
    print("-" * 70)
    
    gt = GroundTruthProcess(seed=42, epistasis=eps)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)
    
    # Generate data
    data = gt.generate_data(8000, lambda_param=1.0, min_mutations=0)
    
    # Count which transitions are observed
    observed = set()
    for start_seq, end_seq, b in data:
        i = seq_to_idx(tuple(start_seq.tolist()))
        j = seq_to_idx(tuple(end_seq.tolist()))
        if i != j and hamming(SEQS[i], SEQS[j]) == 1:
            observed.add((i, j))
    
    # Count total Hamming-1 transitions
    total_h1 = sum(1 for i in range(N_STATES) for j in range(N_STATES) 
                   if i != j and hamming(SEQS[i], SEQS[j]) == 1)
    
    print(f"Observed transitions: {len(observed)}/{total_h1} ({len(observed)/total_h1:.1%})")
    
    # Measure error on observed vs unobserved
    observed_error = 0.0
    unobserved_error = 0.0
    n_observed = 0
    n_unobserved = 0
    
    # Train a simple model
    print("Training Full MLE model...")
    model_full, _ = train_full_mle_model(
        gt, data,
        epochs=200,
        lr=0.01,
        bucket_decimals=2,
        early_stop_patience=30,
        early_stop_tolerance=0.001,
        verbose=False
    )
    
    Q_model = model_full.build_Q()
    
    # Calculate error split by observed/unobserved
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                true_rate = Q_true[i, j].item()
                pred_rate = Q_model[i, j].item()
                error = abs(true_rate - pred_rate)
                
                if (i, j) in observed:
                    observed_error += error ** 2
                    n_observed += 1
                else:
                    unobserved_error += error ** 2
                    n_unobserved += 1
    
    observed_rmse = np.sqrt(observed_error / n_observed) if n_observed > 0 else 0
    unobserved_rmse = np.sqrt(unobserved_error / n_unobserved) if n_unobserved > 0 else 0
    
    print(f"\nRMSE on observed transitions ({n_observed}):   {observed_rmse:.4f}")
    print(f"RMSE on unobserved transitions ({n_unobserved}): {unobserved_rmse:.4f}")
    print(f"Ratio (unobserved/observed): {unobserved_rmse/observed_rmse:.2f}x worse")
    
    # Overall error
    total_error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)
    print(f"\nOverall relative Frobenius error: {total_error.item():.4f}")

