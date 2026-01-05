"""
Run Experiment 0 with optimized configuration:
- L=3 (64 states)
- Factorized model: 10k samples (cached from previous run)
- Full MLE model: 50k samples + L2 regularization (weight_decay=0.0001)
- 3 replicates for statistical robustness
- 4 epistasis values: [0.0, 0.3, 0.7, 1.0]
- Compare full MLE (gradient descent) vs factorized model
- Expected behavior: At epistasis=0, both models should achieve similar error (~0.075).
  As epistasis increases, factorized model error should increase while
  full MLE error remains constant.
"""

from main import run_experiment_0_with_replicates

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXPERIMENT 0: Epistasis Sensitivity Analysis (FINAL)")
    print("Configuration:")
    print("  - Sequence length: L=3 (64 states)")
    print("  - Factorized model: 10k samples (loaded from cache)")
    print("  - Full MLE model: 50k samples + L2 regularization (0.0001)")
    print("  - 3 replicates with different seeds")
    print("  - 4 epistasis values: [0.0, 0.3, 0.7, 1.0]")
    print("  - Expected time: ~30-40 minutes (Full MLE training only)")
    print("=" * 70)

    results = run_experiment_0_with_replicates(
        epistasis_levels=[1, 0.75, 0.5, 0.25, 0.0],
        # epistasis_levels=[0.25, 0.0],
        n_replicates=3,  # 3 replicates for statistics
        lambda_param_factorized=0.5,
        lambda_param_full_mle=0.5,
        n_samples_factorized=2500000,
        n_samples_full_mle=2500000,
        weight_decay_full_mle=0.0,  # L2 regularization for full MLE
        use_cache=True,  # Load cached factorized models
        force_retrain=False,
        use_same_dataset=True,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED!")
    print("=" * 70)
