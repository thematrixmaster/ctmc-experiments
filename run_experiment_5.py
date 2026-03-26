"""
Experiment 5: WAG Model Training on Antibody Evolution Data

Train a 20×20 WAG amino acid substitution rate matrix using L-BFGS optimization.
Model: Q = S × diag(π) where S is symmetric exchangeability matrix, π is stationary distribution.

Data: ~1.6M antibody sequence evolution transitions (parent, child, branch_length)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pickle
import time
from pathlib import Path
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Amino acid alphabet (20 standard amino acids)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AMINO_ACIDS)}
N_AA = 20

# Special characters to mask
SPECIAL_CHARS = {'.', 'X'}

# Data paths
DATA_DIR = Path("./data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Output paths
CHECKPOINT_PATH = "./exp5_wag_checkpoint.pkl"
LOGS_DIR = Path("./logs")
RESULTS_DIR = Path("./results")
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Training hyperparameters
BUCKET_DECIMALS = 2  # Branch length discretization
EARLY_STOP_PATIENCE = 10
EARLY_STOP_TOLERANCE = 1e-4
MAX_EPOCHS = 100
LBFGS_MAX_ITER = 20  # Max iterations per L-BFGS step
LBFGS_HISTORY_SIZE = 100

# Parallelization settings
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 CPU free
USE_PARALLEL_LOADING = False  # Set to True to enable parallel file loading (can cause issues with Ctrl+C)
print(f"CPU workers for data loading: {NUM_WORKERS} (parallel loading: {'enabled' if USE_PARALLEL_LOADING else 'disabled'})")

# ==========================================
# 2. DATA LOADING
# ==========================================

def load_transitions_from_file(file_path):
    """
    Load transitions from a single .txt file.

    Format: Each line (after header) contains: parent_seq child_seq branch_length

    Returns:
        List of tuples: (parent_seq, child_seq, branch_length)
    """
    transitions = []
    with open(file_path, 'r') as f:
        # Skip header line
        header = f.readline()

        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            parent_seq = parts[0]
            child_seq = parts[1]
            branch_length = float(parts[2])

            transitions.append((parent_seq, child_seq, branch_length))

    return transitions


def load_all_transitions(data_dir, parallel=None):
    """Load all transitions from all .txt files in a directory."""
    if parallel is None:
        parallel = USE_PARALLEL_LOADING

    txt_files = sorted(data_dir.glob("*.txt"))

    if not parallel or len(txt_files) == 1:
        # Sequential loading
        all_transitions = []
        for txt_file in txt_files:
            transitions = load_transitions_from_file(txt_file)
            all_transitions.extend(transitions)
            print(f"  Loaded {len(transitions):,} transitions from {txt_file.name}")
        return all_transitions

    # Parallel loading
    all_transitions = []
    print(f"  Loading {len(txt_files)} files in parallel using {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all file loading tasks
        future_to_file = {executor.submit(load_transitions_from_file, f): f for f in txt_files}

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(txt_files), desc="  Loading files"):
            txt_file = future_to_file[future]
            try:
                transitions = future.result()
                all_transitions.extend(transitions)
                print(f"  Loaded {len(transitions):,} transitions from {txt_file.name}")
            except Exception as e:
                print(f"  Error loading {txt_file.name}: {e}")

    return all_transitions


def compute_empirical_pi(transitions):
    """
    Compute empirical amino acid frequencies from training data.

    Args:
        transitions: List of (parent, child, branch_length) tuples

    Returns:
        torch.Tensor: 20-element stationary distribution (normalized frequencies)
    """
    aa_counts = Counter()

    for parent, child, _ in transitions:
        for aa in parent + child:
            # Only count standard amino acids
            if aa in AA_TO_IDX:
                aa_counts[aa] += 1

    # Convert to probability distribution
    total = sum(aa_counts.values())
    pi = torch.zeros(N_AA, dtype=torch.float32)

    for aa, count in aa_counts.items():
        pi[AA_TO_IDX[aa]] = count / total

    # Normalize (should already sum to 1, but ensure numerical stability)
    pi = pi / pi.sum()

    return pi


def encode_sequence(seq):
    """
    Encode sequence as tensor of indices.

    Returns:
        indices: Tensor of shape (L,) with AA indices, or -1 for special chars
    """
    indices = []
    for aa in seq:
        if aa in AA_TO_IDX:
            indices.append(AA_TO_IDX[aa])
        else:
            indices.append(-1)  # Mask special characters

    return torch.tensor(indices, dtype=torch.long)


class TransitionDataset:
    """Dataset for sequence transitions with masking."""

    def __init__(self, transitions, pi):
        """
        Args:
            transitions: List of (parent_seq, child_seq, branch_length)
            pi: Stationary distribution (for reference, not used directly)
        """
        self.transitions = transitions
        self.pi = pi

        # Pre-encode all sequences
        self.encoded_data = []
        print(f"  Encoding {len(transitions):,} sequences...")
        for i, (parent, child, b) in enumerate(transitions):
            if i % 100000 == 0 and i > 0:
                print(f"    Encoded {i:,}/{len(transitions):,} sequences...")

            parent_enc = encode_sequence(parent)
            child_enc = encode_sequence(child)

            # Create mask: True for valid sites, False for sites with special chars
            mask = (parent_enc >= 0) & (child_enc >= 0)

            self.encoded_data.append({
                'parent': parent_enc,
                'child': child_enc,
                'branch_length': b,
                'mask': mask
            })
        print(f"  Finished encoding {len(transitions):,} sequences")

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def to_tensors(self, device=DEVICE):
        """
        Convert all data to tensors on device.

        Returns:
            Dict with padded tensors and metadata
        """
        # Find max sequence length
        max_len = max(len(d['parent']) for d in self.encoded_data)

        n = len(self.encoded_data)
        parents = torch.full((n, max_len), -1, dtype=torch.long)
        children = torch.full((n, max_len), -1, dtype=torch.long)
        masks = torch.zeros((n, max_len), dtype=torch.bool)
        branch_lengths = torch.zeros(n, dtype=torch.float32)

        for i, data in enumerate(self.encoded_data):
            L = len(data['parent'])
            parents[i, :L] = data['parent']
            children[i, :L] = data['child']
            masks[i, :L] = data['mask']
            branch_lengths[i] = data['branch_length']

        return {
            'parents': parents.to(device),
            'children': children.to(device),
            'masks': masks.to(device),
            'branch_lengths': branch_lengths.to(device)
        }


# ==========================================
# 3. WAG MODEL
# ==========================================

class WAGModel(nn.Module):
    """
    WAG model: Q = S × diag(π)

    S is a symmetric 20×20 exchangeability matrix (190 parameters).
    π is the fixed stationary distribution.
    """

    def __init__(self, pi):
        """
        Args:
            pi: Stationary distribution (20-element tensor)
        """
        super().__init__()

        # Store stationary distribution (not trainable)
        self.register_buffer('pi', pi)

        # Parameterize upper triangle of S (symmetric exchangeability)
        # Number of parameters: (20 * 19) / 2 = 190
        self.log_S_upper = nn.Parameter(torch.randn(190) * 0.01)

        print(f"  WAGModel: 190 trainable parameters (symmetric S)")
        print(f"  Stationary distribution π fixed to empirical frequencies")

    def build_S(self):
        """Build symmetric exchangeability matrix S from parameters."""
        S = torch.zeros((N_AA, N_AA), device=self.log_S_upper.device)

        # Fill upper triangle with softplus(log_S) for positivity
        idx = 0
        for i in range(N_AA):
            for j in range(i + 1, N_AA):
                rate = torch.nn.functional.softplus(self.log_S_upper[idx])
                S[i, j] = rate
                S[j, i] = rate  # Symmetric
                idx += 1

        return S

    def build_Q(self):
        """
        Build rate matrix Q = S × diag(π).

        Ensures Q satisfies CTMC constraints:
        - Off-diagonal Q[i,j] = S[i,j] * π[j] ≥ 0
        - Row sums = 0 (diagonal = -row_sum)
        """
        S = self.build_S()

        # Q[i,j] = S[i,j] * π[j] for i ≠ j
        Q = S * self.pi.unsqueeze(0)  # Broadcasting: (20,20) * (1,20) = (20,20)

        # Set diagonal to zero temporarily
        Q = Q * (1.0 - torch.eye(N_AA, device=Q.device))

        # Set diagonal to -row_sum
        row_sums = Q.sum(dim=1)
        Q = Q - torch.diag(row_sums)

        return Q


# ==========================================
# 4. LIKELIHOOD COMPUTATION
# ==========================================

def compute_log_likelihood(model, data_tensors, bucket_decimals=BUCKET_DECIMALS, show_progress=False, desc="Computing likelihood"):
    """
    Compute log-likelihood of data given model using branch length bucketing.

    Args:
        model: WAGModel instance
        data_tensors: Dict from dataset.to_tensors()
        bucket_decimals: Decimals for branch length discretization
        show_progress: Whether to show progress bar
        desc: Description for progress bar

    Returns:
        log_likelihood: Total log P(data | Q)
        n_valid_sites: Total number of valid (non-masked) sites
    """
    parents = data_tensors['parents']
    children = data_tensors['children']
    masks = data_tensors['masks']
    branch_lengths = data_tensors['branch_lengths']

    # Build Q matrix
    Q = model.build_Q()

    # Bucket branch lengths
    bucket_factor = 10 ** bucket_decimals
    branch_lengths_bucketed = torch.round(branch_lengths * bucket_factor) / bucket_factor
    unique_branch_lengths = torch.unique(branch_lengths_bucketed)

    n_buckets = len(unique_branch_lengths)

    # Compute log-likelihood by bucketing
    total_log_likelihood = 0.0
    total_valid_sites = 0

    # Create iterator with optional progress bar (leave=True to keep progress visible)
    iterator = tqdm(unique_branch_lengths, desc=desc, disable=not show_progress,
                    total=n_buckets, unit="bucket", leave=True, ncols=100)

    for b_val in iterator:
        # Find all transitions with this branch length
        batch_mask = (branch_lengths_bucketed == b_val)

        if batch_mask.sum() == 0:
            continue

        # Compute transition probability matrix P(t) = exp(t * Q)
        P = torch.matrix_exp(b_val * Q)  # (20, 20)

        # Get batch data
        batch_parents = parents[batch_mask]  # (batch_size, max_len)
        batch_children = children[batch_mask]
        batch_masks = masks[batch_mask]

        # Vectorized computation over all transitions and sites
        # Get transition probabilities for all sites at once
        # P[parent_aa, child_aa] for all (batch, site) pairs

        # For valid sites, index into P; for invalid sites, set prob = 1 (log_prob = 0)
        # This way masked sites don't contribute to likelihood

        # Initialize with ones (so log(1) = 0 for masked sites)
        probs = torch.ones_like(batch_parents, dtype=torch.float32)

        # For valid sites, get actual transition probabilities
        # batch_parents and batch_children have shape (batch_size, max_len)
        # We need to handle the case where indices might be -1 (masked)

        # Create a safe version that won't index out of bounds
        safe_parents = torch.clamp(batch_parents, min=0, max=N_AA-1)
        safe_children = torch.clamp(batch_children, min=0, max=N_AA-1)

        # Index into P: P[safe_parents, safe_children] gives (batch_size, max_len)
        batch_probs = P[safe_parents, safe_children]

        # Set to 1.0 for masked sites (so log(1) = 0)
        batch_probs = torch.where(batch_masks, batch_probs, torch.ones_like(batch_probs))

        # Compute log probabilities (add epsilon for numerical stability)
        log_probs = torch.log(batch_probs + 1e-10)

        # Sum only over valid sites
        total_log_likelihood += (log_probs * batch_masks.float()).sum()
        total_valid_sites += batch_masks.sum().item()

    return total_log_likelihood, total_valid_sites


# ==========================================
# 5. TRAINING
# ==========================================

def train_wag_model(model, train_dataset, val_dataset, checkpoint_path=CHECKPOINT_PATH):
    """
    Train WAG model using L-BFGS with early stopping.

    Args:
        model: WAGModel instance
        train_dataset: TransitionDataset for training
        val_dataset: TransitionDataset for validation
        checkpoint_path: Path to save/load checkpoint

    Returns:
        model: Trained model
        history: Dict with training history
    """
    # Convert datasets to tensors once
    print("\nConverting datasets to tensors...")
    train_tensors = train_dataset.to_tensors(DEVICE)
    val_tensors = val_dataset.to_tensors(DEVICE)
    print(f"  Train: {len(train_dataset):,} transitions")
    print(f"  Val: {len(val_dataset):,} transitions")

    # Load checkpoint if exists
    start_epoch = 0
    history = {'epochs': [], 'train_nll': [], 'val_nll': [], 'epoch_times': []}
    best_val_nll = float('inf')
    patience_counter = 0
    best_model_state = None

    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_nll = checkpoint['best_val_nll']
        patience_counter = checkpoint['patience_counter']
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best val NLL: {best_val_nll:.4f}")

    # Setup optimizer
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn='strong_wolfe'
    )

    print(f"\nTraining configuration:")
    print(f"  Optimizer: L-BFGS (max_iter={LBFGS_MAX_ITER}, history_size={LBFGS_HISTORY_SIZE})")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Early stopping: patience={EARLY_STOP_PATIENCE}, tolerance={EARLY_STOP_TOLERANCE}")
    print(f"  Branch length buckets: {BUCKET_DECIMALS} decimals")

    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    for epoch in range(start_epoch, MAX_EPOCHS):
        print(f"\n[Epoch {epoch}/{MAX_EPOCHS}]")
        epoch_start = time.time()

        # Closure for L-BFGS
        def closure():
            optimizer.zero_grad()
            log_likelihood, n_sites = compute_log_likelihood(
                model, train_tensors, BUCKET_DECIMALS,
                show_progress=True, desc=f"Epoch {epoch} [Train]"
            )
            nll = -log_likelihood / n_sites  # Negative log-likelihood per site
            nll.backward()
            return nll

        # L-BFGS step (will call closure multiple times for line search)
        print("  Training (L-BFGS may call closure multiple times for line search)...")
        train_nll = optimizer.step(closure).item()

        # Compute validation NLL
        print("  Computing validation NLL...")
        with torch.no_grad():
            val_log_likelihood, val_n_sites = compute_log_likelihood(
                model, val_tensors, BUCKET_DECIMALS,
                show_progress=True, desc=f"Epoch {epoch} [Val]"
            )
            val_nll = -val_log_likelihood.item() / val_n_sites

        epoch_time = time.time() - epoch_start
        print()  # Blank line for readability

        # Record history
        history['epochs'].append(epoch)
        history['train_nll'].append(train_nll)
        history['val_nll'].append(val_nll)
        history['epoch_times'].append(epoch_time)

        # Early stopping check
        if val_nll < best_val_nll - EARLY_STOP_TOLERANCE:
            best_val_nll = val_nll
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state': best_model_state,
                'history': history,
                'best_val_nll': best_val_nll,
                'patience_counter': patience_counter,
                'pi': model.pi.cpu()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f">>> Epoch {epoch:3d}/{MAX_EPOCHS}: Train NLL={train_nll:.6f}, Val NLL={val_nll:.6f} * [BEST - Saved checkpoint] ({epoch_time:.1f}s)")
        else:
            patience_counter += 1
            print(f">>> Epoch {epoch:3d}/{MAX_EPOCHS}: Train NLL={train_nll:.6f}, Val NLL={val_nll:.6f}   [Patience: {patience_counter}/{EARLY_STOP_PATIENCE}] ({epoch_time:.1f}s)")

        print("-" * 80)  # Separator between epochs

        # Check early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best val NLL: {best_val_nll:.6f}")
            model.load_state_dict(best_model_state)
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# ==========================================
# 6. EVALUATION
# ==========================================

def evaluate_model(model, dataset, dataset_name="Dataset"):
    """
    Evaluate model on a dataset.

    Returns:
        nll: Negative log-likelihood per site
        n_sites: Total number of valid sites
    """
    print(f"\nEvaluating on {dataset_name}...")

    with torch.no_grad():
        data_tensors = dataset.to_tensors(DEVICE)
        log_likelihood, n_sites = compute_log_likelihood(
            model, data_tensors, BUCKET_DECIMALS,
            show_progress=True, desc=f"Evaluating {dataset_name}"
        )
        nll = -log_likelihood.item() / n_sites

    print(f"  {dataset_name} NLL: {nll:.6f} ({n_sites:,} sites)")
    return nll, n_sites


# ==========================================
# 7. MAIN
# ==========================================

def main():
    """Main training and evaluation pipeline."""

    print("=" * 60)
    print("Experiment 5: WAG Model Training")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading training data...")
    train_transitions = load_all_transitions(TRAIN_DIR)
    print(f"  Total: {len(train_transitions):,} transitions")

    print("\n[2/6] Loading validation data...")
    val_transitions = load_all_transitions(VAL_DIR)
    print(f"  Total: {len(val_transitions):,} transitions")

    print("\n[3/6] Loading test data...")
    test_transitions = load_all_transitions(TEST_DIR)
    print(f"  Total: {len(test_transitions):,} transitions")

    # Compute empirical stationary distribution
    print("\n[4/6] Computing empirical stationary distribution π...")
    pi = compute_empirical_pi(train_transitions)
    print(f"  π computed from {len(train_transitions):,} transitions")
    print(f"  Top 5 most frequent AAs:")
    top5_idx = torch.argsort(pi, descending=True)[:5]
    for idx in top5_idx:
        aa = IDX_TO_AA[idx.item()]
        freq = pi[idx].item()
        print(f"    {aa}: {freq:.4f}")

    # Create datasets
    print("\n[5/6] Creating datasets...")
    train_dataset = TransitionDataset(train_transitions, pi)
    val_dataset = TransitionDataset(val_transitions, pi)
    test_dataset = TransitionDataset(test_transitions, pi)
    print(f"  Train dataset: {len(train_dataset):,} transitions")
    print(f"  Val dataset: {len(val_dataset):,} transitions")
    print(f"  Test dataset: {len(test_dataset):,} transitions")

    # Initialize model
    print("\n[6/6] Initializing WAG model...")
    model = WAGModel(pi).to(DEVICE)

    # Train model
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    model, history = train_wag_model(model, train_dataset, val_dataset)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    train_nll, train_sites = evaluate_model(model, train_dataset, "Train")
    val_nll, val_sites = evaluate_model(model, val_dataset, "Validation")
    test_nll, test_sites = evaluate_model(model, test_dataset, "Test")

    # Save final results
    print("\nSaving final results...")

    # Save Q matrix and parameters
    with torch.no_grad():
        Q = model.build_Q().cpu().numpy()
        S = model.build_S().cpu().numpy()
        pi_np = model.pi.cpu().numpy()

    results = {
        'Q': Q,
        'S': S,
        'pi': pi_np,
        'train_nll': train_nll,
        'val_nll': val_nll,
        'test_nll': test_nll,
        'train_sites': train_sites,
        'val_sites': val_sites,
        'test_sites': test_sites,
        'history': history
    }

    results_path = RESULTS_DIR / "exp5_wag_matrix.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved to {results_path}")

    # Save evaluation summary
    eval_path = RESULTS_DIR / "exp5_evaluation.txt"
    with open(eval_path, 'w') as f:
        f.write("Experiment 5: WAG Model Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Train NLL: {train_nll:.6f} ({train_sites:,} sites)\n")
        f.write(f"Val NLL:   {val_nll:.6f} ({val_sites:,} sites)\n")
        f.write(f"Test NLL:  {test_nll:.6f} ({test_sites:,} sites)\n")
        f.write(f"\nTotal epochs: {len(history['epochs'])}\n")
        f.write(f"Best val NLL: {min(history['val_nll']):.6f}\n")
    print(f"  Saved to {eval_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nFinal results:")
    print(f"  Train NLL: {train_nll:.6f}")
    print(f"  Val NLL:   {val_nll:.6f}")
    print(f"  Test NLL:  {test_nll:.6f}")
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"Results: {results_path}")
    print(f"Evaluation: {eval_path}")


if __name__ == "__main__":
    main()
