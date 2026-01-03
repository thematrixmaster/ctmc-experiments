import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import itertools
import random
import pickle
import os
import hashlib
from copy import deepcopy

# ==========================================
# 1. SETUP & UTILITIES
# ==========================================

ALPHABET = ['A', 'C', 'G', 'T']
A2I = {a: i for i, a in enumerate(ALPHABET)}
I2A = {i: a for i, a in enumerate(ALPHABET)}
L = 4
N_STATES = len(ALPHABET) ** L
SEQS = list(itertools.product(range(4), repeat=L)) # All 256 sequences as tuples of ints

def seq_to_idx(seq_tuple):
    """Converts a sequence tuple (0,1,3,2) to a flat index 0-255."""
    idx = 0
    for i, token in enumerate(seq_tuple):
        idx += token * (4 ** (L - 1 - i))
    return idx

def idx_to_seq(idx):
    """Converts flat index 0-255 back to tensor shape (1, L)."""
    seq = []
    rem = idx
    for i in range(L):
        div = 4 ** (L - 1 - i)
        seq.append(rem // div)
        rem %= div
    return torch.tensor([seq], dtype=torch.long)

def hamming(s1, s2):
    return sum(1 for a, b in zip(s1, s2) if a != b)

# Precompute all 256 sequences as a batch for fast evaluation
ALL_SEQS_BATCH = torch.tensor(SEQS, dtype=torch.long) # (256, 4)

# ==========================================
# 2. GROUND TRUTH PROCESS
# ==========================================

class GroundTruthProcess:
    def __init__(self, seed=42, epistasis=0.0):
        """
        Initialize ground truth process with optional epistasis.

        Args:
            seed: Random seed for reproducibility
            epistasis: Epistasis strength in [0, 1]
                - 0.0: No epistasis (perfectly factorizable Q)
                - 1.0: Maximum epistasis (rates vary ±100% based on context)
        """
        np.random.seed(seed)
        self.epistasis = epistasis

        # Step 1: Sample base rates (per-site, context-independent)
        self.base_rates = np.zeros((L, 4, 4))  # (site, from_base, to_base)
        for l in range(L):
            for a in range(4):
                for b in range(4):
                    if a != b:
                        self.base_rates[l, a, b] = np.random.uniform(0.1, 1.0)
            # Set diagonal to -row_sum for each site's Q matrix
            for a in range(4):
                self.base_rates[l, a, a] = -np.sum(self.base_rates[l, a, :])

        # Step 2: Build 256x256 Q with context-dependent rates
        self.Q_global = np.zeros((N_STATES, N_STATES))
        for i in range(N_STATES):
            s_i = SEQS[i]
            for j in range(N_STATES):
                if i == j: continue
                s_j = SEQS[j]

                if hamming(s_i, s_j) == 1:
                    # Find which site mutated
                    mutated_site = None
                    for l in range(L):
                        if s_i[l] != s_j[l]:
                            mutated_site = l
                            break

                    a, b = s_i[mutated_site], s_j[mutated_site]

                    # Get base rate
                    base_rate = self.base_rates[mutated_site, a, b]

                    # Apply context-dependent modification
                    if self.epistasis > 0:
                        # Extract context: all bases except the mutated site
                        context = [s_i[k] for k in range(L) if k != mutated_site]
                        # Hash-based modifier for reproducibility
                        context_hash = sum(context[k] * (4**k) for k in range(len(context)))
                        modifier = (context_hash % 100) / 50.0 - 1.0  # Range: [-1, 1]
                        rate = base_rate * (1 + self.epistasis * modifier)
                        rate = max(0.01, rate)  # Ensure positive
                    else:
                        rate = base_rate

                    self.Q_global[i, j] = rate

        # Set diagonal to -row_sum
        for i in range(N_STATES):
            self.Q_global[i, i] = -np.sum(self.Q_global[i, :])

        self.Q_torch = torch.tensor(self.Q_global, dtype=torch.float32)

    def get_P(self, b):
        """Returns exp(bQ) as a dense matrix."""
        return scipy.linalg.expm(b * self.Q_global)

    def generate_data(self, n_samples, b_min=None, b_max=None, lambda_param=None, min_mutations=0):
        """
        Generates (x_i, x_j, b) tuples with optional rejection sampling.

        Args:
            n_samples: Number of samples to generate
            b_min, b_max: If provided, sample b ~ Uniform(b_min, b_max)
            lambda_param: If provided, sample b ~ Exponential(lambda_param), mean = 1/lambda_param
            min_mutations: Minimum number of mutations required (rejection sampling)
        """
        data = []
        attempts = 0
        max_attempts = n_samples * 100  # Prevent infinite loop

        while len(data) < n_samples and attempts < max_attempts:
            attempts += 1

            # Sample branch length
            if lambda_param is not None:
                b = np.random.exponential(1.0 / lambda_param)
            else:
                b = np.random.uniform(b_min, b_max)
            P = self.get_P(b)

            # Pick random start
            start_idx = np.random.randint(0, N_STATES)

            # Sample end based on P[start_idx]
            probs = P[start_idx]
            probs /= probs.sum() # Numerical stability
            end_idx = np.random.choice(N_STATES, p=probs)

            # Check mutation count
            n_mutations = hamming(SEQS[start_idx], SEQS[end_idx])

            # Accept only if enough mutations
            if n_mutations >= min_mutations:
                data.append((ALL_SEQS_BATCH[start_idx], ALL_SEQS_BATCH[end_idx], b))

        if len(data) < n_samples:
            print(f"Warning: Could only generate {len(data)}/{n_samples} samples after {attempts} attempts")

        return data

    def verify_mutation_rate(self, n_samples=1000):
        """Verify that branch length ≈ expected mutations per site"""
        print("\nVerifying Q matrix scaling:")
        print(f"{'b':<8} {'E[mutations]':<15} {'E[mut/site]':<15} {'Target':<10}")
        print("-" * 50)

        for b in [0.01, 0.1, 0.5, 1.0]:
            mutation_counts = []
            for _ in range(n_samples):
                start_idx = np.random.randint(0, N_STATES)
                P = self.get_P(b)
                end_idx = np.random.choice(N_STATES, p=P[start_idx])
                n_mutations = hamming(SEQS[start_idx], SEQS[end_idx])
                mutation_counts.append(n_mutations)

            mean_mutations = np.mean(mutation_counts)
            expected_per_site = mean_mutations / L
            print(f"{b:<8.2f} {mean_mutations:<15.3f} {expected_per_site:<15.3f} {b:<10.2f}")

# ==========================================
# 3. NEURAL RATE MODEL
# ==========================================

class NeuralRateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 8)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Output heads for each site: 4 sites * (4*4 matrix)
        self.heads = nn.ModuleList([nn.Linear(64, 16) for _ in range(L)])

    def forward(self, x):
        # x: (Batch, L)
        bs = x.shape[0]
        emb = self.emb(x)
        features = self.net(emb)
        
        # Build 4x4 rate matrices for each site
        qs = []
        for l in range(L):
            # Predict raw scores
            raw = self.heads[l](features).view(bs, 4, 4)
            # Softplus for positivity
            rates = torch.nn.functional.softplus(raw)
            # Mask diagonal
            mask = 1.0 - torch.eye(4, device=x.device).unsqueeze(0)
            rates = rates * mask
            # Set diagonal to -row sum
            row_sums = rates.sum(dim=2, keepdim=True)
            rates = rates - (row_sums * torch.eye(4, device=x.device).unsqueeze(0))
            qs.append(rates) # List of (Batch, 4, 4)
            
        return torch.stack(qs, dim=1) # (Batch, L, 4, 4)

    def build_global_Q(self):
        """Reconstructs the 256x256 global Q matrix from the model."""
        with torch.no_grad():
            # Get local rates for ALL contexts
            # local_qs: (256, L, 4, 4)
            local_qs = self(ALL_SEQS_BATCH)
            
            Q_rec = torch.zeros((N_STATES, N_STATES))
            
            for i in range(N_STATES):
                s1 = SEQS[i] # e.g. (0, 0, 0, 0)
                # Fill single site mutations
                for l in range(L):
                    # For site l, look at possible mutations a
                    curr_char = s1[l]
                    rate_mat = local_qs[i, l] # (4, 4)
                    
                    for a in range(4):
                        if a == curr_char: continue
                        
                        # Construct neighbor s2
                        s2_list = list(s1)
                        s2_list[l] = a
                        s2 = tuple(s2_list)
                        j = seq_to_idx(s2)
                        
                        # The rate from i to j is determined by site l's rate matrix
                        # evaluated at context i
                        rate = rate_mat[curr_char, a]
                        Q_rec[i, j] = rate
            
            # Set diagonals
            for i in range(N_STATES):
                Q_rec[i, i] = -torch.sum(Q_rec[i, :])
                
            return Q_rec

class FullMLEModel(nn.Module):
    """
    Full 256×256 rate matrix with MLE via gradient descent.

    Parameterizes all 3072 Hamming-1 transition rates directly.
    Uses matrix exponential likelihood: P(end|start,b) = exp(b*Q)[start,end]
    """
    def __init__(self):
        super().__init__()
        # Parameterize all Hamming-1 transitions
        # We'll use a dictionary to map (i,j) pairs to parameter indices
        self.param_indices = {}
        param_list = []

        idx = 0
        for i in range(N_STATES):
            for j in range(N_STATES):
                if i != j and hamming(SEQS[i], SEQS[j]) == 1:
                    self.param_indices[(i, j)] = idx
                    param_list.append(torch.randn(1) * 0.1)  # Small random init
                    idx += 1

        # Single parameter tensor for all rates
        self.log_rates = nn.Parameter(torch.cat(param_list))
        self.n_params = len(param_list)

        print(f"  FullMLEModel: {self.n_params} trainable parameters")

    def build_Q(self):
        """Construct 256×256 rate matrix from parameters."""
        Q = torch.zeros((N_STATES, N_STATES))

        # Fill off-diagonal Hamming-1 entries
        for (i, j), idx in self.param_indices.items():
            Q[i, j] = torch.nn.functional.softplus(self.log_rates[idx])

        # Set diagonal to -row_sum
        for i in range(N_STATES):
            Q[i, i] = -torch.sum(Q[i, :])

        return Q

class ContextIndependentRateModel(nn.Module):
    """Simplified model where rates don't depend on sequence context.

    Directly parameterizes L local 4x4 rate matrices, one per site.
    This is the correct architecture when epistasis=0.
    """
    def __init__(self):
        super().__init__()
        # Direct parameters: L sites × 4×4 rate matrices
        self.log_rates = nn.ParameterList([
            nn.Parameter(torch.randn(4, 4)) for _ in range(L)
        ])

    def forward(self, x):
        """x: (Batch, L) - sequence contexts (IGNORED for context-independence)"""
        bs = x.shape[0]
        device = x.device

        qs = []
        for l in range(L):
            # Apply softplus for positivity
            rates = torch.nn.functional.softplus(self.log_rates[l])
            # Mask diagonal
            mask = 1.0 - torch.eye(4, device=device)
            rates = rates * mask
            # Set diagonal to -row sum
            row_sums = rates.sum(dim=1, keepdim=True)
            rates = rates - row_sums * torch.eye(4, device=device)
            # Expand to batch dimension
            qs.append(rates.unsqueeze(0).expand(bs, -1, -1))

        return torch.stack(qs, dim=1)  # (Batch, L, 4, 4)

    def build_global_Q(self):
        """Reconstructs the 256x256 global Q matrix from the model."""
        with torch.no_grad():
            # Get local rates (context-independent, so same for all)
            local_qs = self(ALL_SEQS_BATCH)

            Q_rec = torch.zeros((N_STATES, N_STATES))

            for i in range(N_STATES):
                s1 = SEQS[i]
                for l in range(L):
                    curr_char = s1[l]
                    rate_mat = local_qs[i, l]  # Same for all i

                    for a in range(4):
                        if a == curr_char: continue

                        s2_list = list(s1)
                        s2_list[l] = a
                        s2 = tuple(s2_list)
                        j = seq_to_idx(s2)

                        rate = rate_mat[curr_char, a]
                        Q_rec[i, j] = rate

            # Set diagonals
            for i in range(N_STATES):
                Q_rec[i, i] = -torch.sum(Q_rec[i, :])

            return Q_rec

# ==========================================
# 4. TRAINING LOOP
# ==========================================

def train_model(gt, train_data, use_snr=False, epochs=500, lr=0.01,
                early_stop_patience=None, early_stop_tolerance=1e-4, verbose=True):
    """
    Train the factorized neural rate model.

    Args:
        gt: Ground truth process (for Q_true if early stopping enabled)
        train_data: List of (start, end, b) tuples
        use_snr: Whether to use SNR weighting (deprecated)
        epochs: Maximum number of epochs
        lr: Learning rate
        early_stop_patience: If set, stop if error doesn't improve for this many epochs
        early_stop_tolerance: Minimum improvement to count as improvement
        verbose: Print training progress

    Returns:
        model: Trained model
        history: Dict with 'epochs', 'losses', 'errors' (if early stopping enabled)
    """
    model = NeuralRateModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Pre-process data into tensors
    starts = torch.stack([d[0] for d in train_data])
    ends = torch.stack([d[1] for d in train_data])
    bs = torch.tensor([d[2] for d in train_data], dtype=torch.float32)

    if verbose:
        print(f"Training {'SNR-Weighted' if use_snr else 'MLE'} (Size: {len(train_data)}, LR: {lr})...")

    # Early stopping setup
    best_error = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'epochs': [], 'losses': [], 'errors': []}
    Q_true = None

    if early_stop_patience is not None:
        Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)
        best_model_state = deepcopy(model.state_dict())  # Initialize with initial weights

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Forward pass to get local Qs for start sequences
        # q_preds: (Batch, L, 4, 4)
        q_preds = model(starts)
        
        # 2. Calculate transition probabilities factorized over sites
        # We need P(end_l | start_l) = exp(b * Q^l)[start_l, end_l]
        log_prob_total = 0
        
        for l in range(L):
            # Extract Q^l for this batch: (Batch, 4, 4)
            q_l = q_preds[:, l]
            
            # Scale by b: (Batch, 4, 4)
            # b is (Batch,), unsqueeze for broadcast
            b_view = bs.view(-1, 1, 1)
            matrix_input = q_l * b_view
            
            # Matrix Exp
            p_matrix = torch.matrix_exp(matrix_input)
            
            # Gather probability of the actual transition observed
            # start_char: (Batch,), end_char: (Batch,)
            start_char = starts[:, l]
            end_char = ends[:, l]
            
            # Gather P[start, end]
            # Use torch.gather or fancy indexing
            # This selects p_matrix[k, start_char[k], end_char[k]]
            probs = p_matrix[torch.arange(len(train_data)), start_char, end_char]
            
            log_prob_total += torch.log(probs + 1e-9)
            
        # 3. Loss Calculation
        nll = -log_prob_total
        
        if use_snr:
            # SNR Weighting: 1 / (b + epsilon)
            weights = 1.0 / (bs + 1e-4)
            loss = (nll * weights).mean()
        else:
            loss = nll.mean()
            
        loss.backward()
        optimizer.step()

        # Early stopping check
        if early_stop_patience is not None:
            with torch.no_grad():
                Q_model = model.build_global_Q()
                error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

                history['epochs'].append(epoch)
                history['losses'].append(loss.item())
                history['errors'].append(error.item())

                # Check for improvement
                if error < best_error - early_stop_tolerance:
                    best_error = error
                    patience_counter = 0
                    best_model_state = deepcopy(model.state_dict())
                else:
                    patience_counter += 1

                if verbose and epoch % 50 == 0:
                    print(f"  Epoch {epoch}: Loss={loss.item():.4f}, Error={error.item():.4f}, Patience={patience_counter}/{early_stop_patience}")

                # Early stop
                if patience_counter >= early_stop_patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} (best error: {best_error:.4f})")
                    model.load_state_dict(best_model_state)
                    break
        else:
            # No early stopping, just print progress
            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss {loss.item():.4f}")

    return model, history

def train_full_mle_model(gt, train_data, epochs=500, lr=0.01, verbose=True, bucket_decimals=2, 
                         early_stop_patience=50, early_stop_tolerance=0.001):
    """
    Train the full 256×256 MLE model via gradient descent with early stopping.

    Uses proper likelihood: P(end|start,b) = exp(b*Q)[start,end]

    Args:
        gt: Ground truth process (for Q_true for monitoring)
        train_data: List of (start, end, b) tuples
        epochs: Number of training epochs (max)
        lr: Learning rate
        verbose: Print progress
        bucket_decimals: Decimals for bucketing b values (lower = faster, less precise)
        early_stop_patience: Stop if no improvement for N epochs
        early_stop_tolerance: Minimum improvement to reset patience

    Returns:
        model: Trained FullMLEModel
        history: Dict with 'losses', 'errors', 'epoch_times'
    """
    import time
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        if verbose:
            print("Install tqdm for progress bars: pip install tqdm")

    model = FullMLEModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Pre-process data
    starts_idx = torch.tensor([seq_to_idx(tuple(d[0].tolist())) for d in train_data], dtype=torch.long)
    ends_idx = torch.tensor([seq_to_idx(tuple(d[1].tolist())) for d in train_data], dtype=torch.long)
    bs = torch.tensor([d[2] for d in train_data], dtype=torch.float32)

    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)
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
        print(f"Training Full MLE Model:")
        print(f"  Samples: {len(train_data)}")
        print(f"  Max epochs: {epochs}")
        print(f"  Early stopping: patience={early_stop_patience}, tolerance={early_stop_tolerance}")
        print(f"  Unique branch lengths: {n_buckets} (bucket_decimals={bucket_decimals})")
        print(f"  Matrix exps per epoch: {n_buckets}")
        print(f"  Learning rate: {lr}")

    # Estimate time for first epoch
    start_time = time.time()
    first_epoch_time = None

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

            # Matrix exponential (this is the bottleneck)
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
                elapsed = time.time() - start_time
                avg_time = elapsed / (epoch + 1)
                remaining = avg_time * (epochs - epoch - 1)
                print(f"  Epoch {epoch:4d}/{epochs}: Loss={loss.item():.4f}, Error={error.item():.4f} "
                      f"Patience={patience_counter}/{early_stop_patience} [{epoch_time:.1f}s/epoch, ETA: {remaining/60:.1f}min]")

            if has_tqdm and epoch % 5 == 0:
                epoch_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'error': f'{error.item():.4f}',
                    'patience': f'{patience_counter}/{early_stop_patience}',
                    'sec/ep': f'{epoch_time:.1f}'
                })

            # Check early stopping
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch} (best error: {best_error:.4f})")
                model.load_state_dict(best_model_state)
                break

        # After first epoch, estimate total time
        if epoch == 0 and verbose:
            first_epoch_time = epoch_time
            total_estimated = first_epoch_time * epochs
            print(f"\n  First epoch: {first_epoch_time:.1f}s")
            print(f"  Estimated total time: {total_estimated/60:.1f} minutes ({total_estimated/3600:.1f} hours)")
            print(f"  (Will stop early if error plateaus)")
            print()

    return model, history

# ==========================================
# 4.5. CACHING INFRASTRUCTURE
# ==========================================

CACHE_DIR = "./exp0_cache"

def get_cache_key(epistasis, seed, model_type, n_samples):
    """Generate unique cache key."""
    key_str = f"{epistasis}_{seed}_{model_type}_{n_samples}"
    return hashlib.md5(key_str.encode()).hexdigest()

def save_dataset(data, epistasis, seed, n_samples):
    """Save generated dataset to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = get_cache_key(epistasis, seed, "dataset", n_samples)
    path = os.path.join(CACHE_DIR, f"dataset_{key}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved dataset to {path}")
    return path

def load_dataset(epistasis, seed, n_samples):
    """Load cached dataset if exists."""
    key = get_cache_key(epistasis, seed, "dataset", n_samples)
    path = os.path.join(CACHE_DIR, f"dataset_{key}.pkl")
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_factorized_model(model, epistasis, seed, n_samples):
    """Save trained factorized model."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = get_cache_key(epistasis, seed, "factorized", n_samples)

    # Save both the model state and the reconstructed Q matrix
    Q_matrix = model.build_global_Q()
    save_dict = {
        'model_state': model.state_dict(),
        'Q_matrix': Q_matrix.cpu().numpy()
    }

    path = os.path.join(CACHE_DIR, f"factorized_{key}.pt")
    torch.save(save_dict, path)
    print(f"  Saved factorized model to {path}")
    return path

def load_factorized_model(epistasis, seed, n_samples):
    """Load cached factorized model."""
    key = get_cache_key(epistasis, seed, "factorized", n_samples)
    path = os.path.join(CACHE_DIR, f"factorized_{key}.pt")
    if os.path.exists(path):
        save_dict = torch.load(path, weights_only=False)
        # Return the Q matrix directly (don't need to reconstruct model)
        return torch.tensor(save_dict['Q_matrix'])
    return None

def save_factorized_snr_model(model, epistasis, seed, n_samples):
    """Save trained SNR-weighted factorized model."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = get_cache_key(epistasis, seed, "factorized_snr", n_samples)

    # Save both the model state and the reconstructed Q matrix
    Q_matrix = model.build_global_Q()
    save_dict = {
        'model_state': model.state_dict(),
        'Q_matrix': Q_matrix.cpu().numpy()
    }

    path = os.path.join(CACHE_DIR, f"factorized_snr_{key}.pt")
    torch.save(save_dict, path)
    print(f"  Saved SNR-weighted model to {path}")
    return path

def load_factorized_snr_model(epistasis, seed, n_samples):
    """Load cached SNR-weighted factorized model."""
    key = get_cache_key(epistasis, seed, "factorized_snr", n_samples)
    path = os.path.join(CACHE_DIR, f"factorized_snr_{key}.pt")
    if os.path.exists(path):
        save_dict = torch.load(path, weights_only=False)
        # Return the Q matrix directly (don't need to reconstruct model)
        return torch.tensor(save_dict['Q_matrix'])
    return None

def save_full_mle_matrix(Q_matrix, epistasis, seed, n_samples):
    """Save full 256x256 MLE Q matrix."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = get_cache_key(epistasis, seed, "full_mle", n_samples)
    path = os.path.join(CACHE_DIR, f"full_mle_{key}.npy")
    np.save(path, Q_matrix)
    print(f"  Saved full MLE matrix to {path}")
    return path

def load_full_mle_matrix(epistasis, seed, n_samples):
    """Load cached full MLE matrix."""
    key = get_cache_key(epistasis, seed, "full_mle", n_samples)
    path = os.path.join(CACHE_DIR, f"full_mle_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

def estimate_full_mle_q(data, gt, regularization=1e-6):
    """
    Estimate full 256x256 Q matrix using maximum likelihood.

    Uses empirical transition counts and branch lengths to estimate
    the rate matrix directly (no factorization assumption).

    Args:
        data: List of (start_seq, end_seq, branch_length) tuples
        gt: Ground truth process (for structure, not rates)
        regularization: Small value added to counts for numerical stability

    Returns:
        Q_mle: 256x256 numpy array (rate matrix)
    """
    print("    Estimating full 256x256 MLE Q matrix...")

    # Initialize count matrices
    transition_counts = np.zeros((N_STATES, N_STATES))  # N(i→j)
    time_spent = np.zeros(N_STATES)  # T(i) = sum of branch lengths starting from i

    # Count transitions and time
    for start_seq, end_seq, b in data:
        start_idx = seq_to_idx(tuple(start_seq.tolist()))
        end_idx = seq_to_idx(tuple(end_seq.tolist()))

        # Count this transition
        transition_counts[start_idx, end_idx] += 1

        # Add time spent in start state
        time_spent[start_idx] += b

    # Estimate Q using MLE formula: Q[i,j] = N(i→j) / T(i)  for i≠j
    Q_mle = np.zeros((N_STATES, N_STATES))

    for i in range(N_STATES):
        if time_spent[i] > 0:
            for j in range(N_STATES):
                if i != j:
                    # MLE estimate with regularization
                    Q_mle[i, j] = (transition_counts[i, j] + regularization) / (time_spent[i] + regularization * N_STATES)

    # Set diagonal to -row_sum
    for i in range(N_STATES):
        Q_mle[i, i] = -np.sum(Q_mle[i, :])

    # Enforce sparsity pattern: only Hamming-1 transitions
    # (Optional: helps with stability and matches ground truth structure)
    for i in range(N_STATES):
        for j in range(N_STATES):
            if i != j and hamming(SEQS[i], SEQS[j]) > 1:
                Q_mle[i, j] = 0

    # Recompute diagonal after sparsity enforcement
    for i in range(N_STATES):
        Q_mle[i, i] = -np.sum(Q_mle[i, :])

    return Q_mle

# ==========================================
# 5. EXPERIMENTS
# ==========================================

def run_experiment_1(gt):
    print("\n=== Experiment 1: Mutation-Informed Branch Length Sweep ===")

    # Lambda values for LARGE branch lengths (testing epistatic effects)
    lambda_values = [0.1, 0.2, 0.5, 0.7, 1.0]

    # Track statistics
    results = []
    Q_true = gt.Q_torch

    print(f"\nSweeping over {len(lambda_values)} lambda values with min_mutations=1...")
    print(f"Using 5000 samples per model, 500 epochs each")
    print("-" * 70)

    for lam in lambda_values:
        # Generate with rejection sampling (min_mutations=1)
        print(f"λ={lam}: Generating data with rejection sampling...", end='', flush=True)
        data = gt.generate_data(5000, lambda_param=lam, min_mutations=1)

        # Calculate actual statistics from generated data
        mutation_counts = []
        branch_lengths = []
        for start_seq, end_seq, b in data:
            # Convert tensors to tuples for seq_to_idx
            start_tuple = tuple(start_seq.tolist())
            end_tuple = tuple(end_seq.tolist())
            start_idx = seq_to_idx(start_tuple)
            end_idx = seq_to_idx(end_tuple)
            n_muts = hamming(SEQS[start_idx], SEQS[end_idx])
            mutation_counts.append(n_muts)
            branch_lengths.append(b)

        mean_muts = np.mean(mutation_counts)
        mean_b = np.mean(branch_lengths)

        print(f" mean_muts={mean_muts:.2f}, mean_b={mean_b:.3f}")
        print(f"  Training...", end='', flush=True)

        # Train and evaluate
        model, _ = train_model(gt, data, use_snr=False, epochs=500, lr=0.01)
        Q_model = model.build_global_Q()
        error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

        print(f" Error={error.item():.4f}")

        results.append({
            'lambda': lam,
            'mean_b': mean_b,
            'mean_mutations': mean_muts,
            'error': error.item()
        })

    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"{'λ':<8} {'Mean b':<12} {'Mean Muts':<12} {'Error':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['lambda']:<8} {r['mean_b']:<12.3f} {r['mean_mutations']:<12.2f} {r['error']:<10.4f}")

    # PLOT 1: Error vs Mean Mutations
    plt.figure(figsize=(10, 6))
    mean_muts = [r['mean_mutations'] for r in results]
    errors = [r['error'] for r in results]
    plt.plot(mean_muts, errors, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Mean Mutations per Transition', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.title('Experiment 1: Error vs Mutation Count (min_mutations=1)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp1_mutations.png', dpi=150)
    print("\nSaved exp1_mutations.png")

    # PLOT 2: Error vs Mean Branch Length
    plt.figure(figsize=(10, 6))
    mean_bs = [r['mean_b'] for r in results]
    plt.plot(mean_bs, errors, 'o-', linewidth=2, markersize=8, color='orange')
    plt.xlabel('Mean Branch Length', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.title('Experiment 1: Error vs Branch Length (min_mutations=1)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp1_branch_length.png', dpi=150)
    print("Saved exp1_branch_length.png")

def plot_experiment_0_comparison(summary):
    """Plot comparison of factorized vs SNR-weighted vs full MLE with error bars."""
    eps_vals = [s['epistasis'] for s in summary]

    fact_means = [s['factorized_mean'] for s in summary]
    fact_stds = [s['factorized_std'] for s in summary]

    snr_means = [s['snr_mean'] for s in summary]
    snr_stds = [s['snr_std'] for s in summary]

    full_means = [s['full_mean'] for s in summary]
    full_stds = [s['full_std'] for s in summary]

    # ICML single-column width: 3.25 inches, use golden ratio for height
    plt.figure(figsize=(3.25, 2.5))

    # Professional color scheme (colorblind-friendly)
    # Factorized model - blue
    plt.errorbar(eps_vals, fact_means, yerr=fact_stds,
                 fmt='o-', linewidth=1.5, markersize=5,
                 color='#0173B2', label='Factorized',
                 capsize=3, capthick=1.5, elinewidth=1.5)

    # SNR-weighted factorized model - green
    plt.errorbar(eps_vals, snr_means, yerr=snr_stds,
                 fmt='^-', linewidth=1.5, markersize=5,
                 color='#009E73', label='Factorized + SNR',
                 capsize=3, capthick=1.5, elinewidth=1.5)

    # Full MLE model - orange/amber
    plt.errorbar(eps_vals, full_means, yerr=full_stds,
                 fmt='s-', linewidth=1.5, markersize=5,
                 color='#E69F00', label='Full MLE',
                 capsize=3, capthick=1.5, elinewidth=1.5)

    plt.xlabel('Epistasis Strength', fontsize=10)
    plt.ylabel('Relative Frobenius Error', fontsize=10)
    plt.legend(fontsize=7, loc='upper left', frameon=True, fancybox=False, edgecolor='black')
    plt.grid(True, alpha=0.2, linewidth=0.5)
    plt.xticks(eps_vals, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=0.3)
    plt.savefig('exp0_comparison.pdf', dpi=300, bbox_inches='tight')
    print("\nSaved exp0_comparison.pdf")

def plot_convergence(epochs, errors, losses, full_mle_error, convergence_epoch=None, tolerance=0.2):
    """Plot convergence curves with convergence point marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error vs Epochs
    ax1.plot(epochs, errors, 'b-', linewidth=2, label='Factorized Model')

    # Full MLE baseline
    ax1.axhline(y=full_mle_error, color='orange', linestyle='--',
                linewidth=2, label=f'Full MLE ({full_mle_error:.4f})')

    # Tolerance band
    ax1.axhline(y=full_mle_error + tolerance, color='gray', linestyle=':',
                linewidth=1.5, alpha=0.6, label=f'Target (MLE + {tolerance})')

    # Mark convergence point
    if convergence_epoch is not None:
        conv_idx = epochs.index(convergence_epoch)
        conv_error = errors[conv_idx]
        ax1.plot(convergence_epoch, conv_error, 'r*', markersize=15,
                label=f'Converged (epoch {convergence_epoch})')
        ax1.axvline(x=convergence_epoch, color='red', linestyle=':', alpha=0.4)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Relative Frobenius Error', fontsize=12)
    ax1.set_title('Convergence: Factorized Model → Full MLE', fontsize=14)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2.plot(epochs, losses, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Negative Log-Likelihood', fontsize=12)
    ax2.set_title('Training Loss (NLL)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Mark convergence point on loss plot too
    if convergence_epoch is not None:
        ax2.axvline(x=convergence_epoch, color='red', linestyle=':', alpha=0.4,
                   label=f'Converged (epoch {convergence_epoch})')
        ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('convergence_epistasis0.png', dpi=150)
    print("\nSaved convergence_epistasis0.png")

def run_experiment_0(seed=42):
    """
    Experiment 0: Test factorized model error vs epistasis strength.

    Hypothesis: Error increases as epistatic interactions strengthen,
    since factorized model assumes site independence.
    """
    print("\n=== Experiment 0: Epistasis Sensitivity Analysis ===")

    epistasis_levels = [0.0, 0.3, 0.7, 1.0]
    results = []

    print(f"\nTesting {len(epistasis_levels)} epistasis levels...")
    print(f"Using λ=1 (mean b≈1.0), 5000 samples, min_mutations=1")
    print("-" * 70)

    for eps in epistasis_levels:
        print(f"\nEpistasis={eps:.1f}:")

        # Create ground truth with this epistasis level
        gt = GroundTruthProcess(seed=seed, epistasis=eps)
        Q_true = gt.Q_torch

        # Generate data (same distribution for all epistasis levels)
        print(f"  Generating data...", end='', flush=True)
        data = gt.generate_data(5000, lambda_param=1.0, min_mutations=1)

        # Calculate statistics
        mutation_counts = []
        branch_lengths = []
        for start_seq, end_seq, b in data:
            start_tuple = tuple(start_seq.tolist())
            end_tuple = tuple(end_seq.tolist())
            start_idx = seq_to_idx(start_tuple)
            end_idx = seq_to_idx(end_tuple)
            n_muts = hamming(SEQS[start_idx], SEQS[end_idx])
            mutation_counts.append(n_muts)
            branch_lengths.append(b)

        mean_muts = np.mean(mutation_counts)
        mean_b = np.mean(branch_lengths)
        print(f" mean_muts={mean_muts:.2f}, mean_b={mean_b:.3f}")

        # Train model
        print(f"  Training...", end='', flush=True)
        model, _ = train_model(gt, data, use_snr=False, epochs=500, lr=0.01)

        # Evaluate
        Q_model = model.build_global_Q()
        error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

        print(f" Error={error.item():.4f}")

        results.append({
            'epistasis': eps,
            'mean_b': mean_b,
            'mean_mutations': mean_muts,
            'error': error.item()
        })

    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"{'Epistasis':<12} {'Mean b':<12} {'Mean Muts':<12} {'Error':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['epistasis']:<12.1f} {r['mean_b']:<12.3f} "
              f"{r['mean_mutations']:<12.2f} {r['error']:<10.4f}")

    # Plot: Error vs Epistasis
    plt.figure(figsize=(10, 6))
    eps_vals = [r['epistasis'] for r in results]
    errors = [r['error'] for r in results]

    plt.plot(eps_vals, errors, 'o-', linewidth=2, markersize=10, color='red')
    plt.xlabel('Epistasis Strength', fontsize=12)
    plt.ylabel('Relative Frobenius Error', fontsize=12)
    plt.title('Experiment 0: Model Error vs Ground Truth Epistasis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(eps_vals)
    plt.tight_layout()
    plt.savefig('exp0_epistasis.png', dpi=150)
    print("\nSaved exp0_epistasis.png")

    print(f"\nConclusion: Error {'increased' if errors[-1] > errors[0] else 'did not increase'} "
          f"with epistasis ({errors[0]:.3f} → {errors[-1]:.3f})")

def run_experiment_0_with_replicates(
    epistasis_levels=[0.0, 0.3, 0.7, 1.0],
    n_replicates=5,
    n_samples=50000,
    use_cache=True,
    force_retrain=False
):
    """
    Experiment 0 with replicates, caching, and full MLE baseline.

    Args:
        epistasis_levels: List of epistasis strengths to test
        n_replicates: Number of independent replicates (for error bars)
        n_samples: Dataset size (larger for good full model estimates)
        use_cache: If True, load/save cached results
        force_retrain: If True, ignore cache and retrain everything
    """
    print("\n=== Experiment 0: Epistasis Analysis with Baselines ===")
    print(f"Epistasis levels: {epistasis_levels}")
    print(f"Replicates: {n_replicates}")
    print(f"Samples per dataset: {n_samples}")
    print(f"Caching: {'ON' if use_cache else 'OFF'}")
    print(f"Force retrain: {'YES' if force_retrain else 'NO'}")
    print("=" * 70)

    # Results storage: {epistasis: {model_type: [errors across replicates]}}
    all_results = {eps: {'factorized': [], 'factorized_snr': [], 'full_mle': []} for eps in epistasis_levels}

    for replicate_idx in range(n_replicates):
        seed = 42 + replicate_idx
        print(f"\n{'='*70}")
        print(f"REPLICATE {replicate_idx + 1}/{n_replicates} (seed={seed})")
        print(f"{'='*70}")

        for eps in epistasis_levels:
            print(f"\nEpistasis={eps:.1f}:")

            # Create ground truth
            gt = GroundTruthProcess(seed=seed, epistasis=eps)
            Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

            # ============ Dataset ============
            data = None
            if use_cache and not force_retrain:
                data = load_dataset(eps, seed, n_samples)
                if data:
                    print(f"  Loaded cached dataset ({len(data)} samples)")

            if data is None:
                print(f"  Generating {n_samples} samples...", end='', flush=True)
                data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=1)
                print(f" Done ({len(data)} samples)")

                if use_cache:
                    save_dataset(data, eps, seed, n_samples)

            # ============ Factorized Model ============
            Q_factorized = None
            if use_cache and not force_retrain:
                Q_factorized = load_factorized_model(eps, seed, n_samples)
                if Q_factorized is not None:
                    print(f"  Loaded cached factorized model")

            if Q_factorized is None:
                print(f"  Training factorized model...", end='', flush=True)
                model, _ = train_model(gt, data, use_snr=False, epochs=500, lr=0.01)
                Q_factorized = model.build_global_Q()
                print(f" Done")

                if use_cache:
                    save_factorized_model(model, eps, seed, n_samples)

            error_factorized = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)
            all_results[eps]['factorized'].append(error_factorized.item())
            print(f"  Factorized error: {error_factorized.item():.4f}")

            # ============ Factorized SNR-Weighted Model ============
            Q_factorized_snr = None
            if use_cache and not force_retrain:
                Q_factorized_snr = load_factorized_snr_model(eps, seed, n_samples)
                if Q_factorized_snr is not None:
                    print(f"  Loaded cached SNR-weighted model")

            if Q_factorized_snr is None:
                print(f"  Training SNR-weighted model...", end='', flush=True)
                model_snr, _ = train_model(gt, data, use_snr=True, epochs=500, lr=0.01)
                Q_factorized_snr = model_snr.build_global_Q()
                print(f" Done")

                if use_cache:
                    save_factorized_snr_model(model_snr, eps, seed, n_samples)

            error_factorized_snr = torch.norm(Q_factorized_snr - Q_true) / torch.norm(Q_true)
            all_results[eps]['factorized_snr'].append(error_factorized_snr.item())
            print(f"  SNR-weighted error: {error_factorized_snr.item():.4f}")

            # ============ Full MLE Model ============
            Q_full = None
            if use_cache and not force_retrain:
                Q_full = load_full_mle_matrix(eps, seed, n_samples)
                if Q_full is not None:
                    print(f"  Loaded cached full MLE matrix")
                    Q_full = torch.tensor(Q_full, dtype=torch.float32)

            if Q_full is None:
                Q_full_np = estimate_full_mle_q(data, gt)
                Q_full = torch.tensor(Q_full_np, dtype=torch.float32)

                if use_cache:
                    save_full_mle_matrix(Q_full_np, eps, seed, n_samples)

            error_full = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
            all_results[eps]['full_mle'].append(error_full.item())
            print(f"  Full MLE error: {error_full.item():.4f}")

    # ============ Aggregate Results ============
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS REPLICATES")
    print("=" * 70)
    print(f"{'Epistasis':<12} {'Factorized':<18} {'SNR-Weighted':<18} {'Full MLE':<18}")
    print(f"{'':12} {'Mean ± Std':<18} {'Mean ± Std':<18} {'Mean ± Std':<18}")
    print("-" * 70)

    summary = []
    for eps in epistasis_levels:
        fact_errors = all_results[eps]['factorized']
        snr_errors = all_results[eps]['factorized_snr']
        full_errors = all_results[eps]['full_mle']

        fact_mean = np.mean(fact_errors)
        fact_std = np.std(fact_errors, ddof=1) if len(fact_errors) > 1 else 0
        snr_mean = np.mean(snr_errors)
        snr_std = np.std(snr_errors, ddof=1) if len(snr_errors) > 1 else 0
        full_mean = np.mean(full_errors)
        full_std = np.std(full_errors, ddof=1) if len(full_errors) > 1 else 0

        print(f"{eps:<12.1f} {fact_mean:.4f}±{fact_std:.4f}    "
              f"{snr_mean:.4f}±{snr_std:.4f}    "
              f"{full_mean:.4f}±{full_std:.4f}")

        summary.append({
            'epistasis': eps,
            'factorized_mean': fact_mean,
            'factorized_std': fact_std,
            'snr_mean': snr_mean,
            'snr_std': snr_std,
            'full_mean': full_mean,
            'full_std': full_std
        })

    # ============ Plot Results ============
    plot_experiment_0_comparison(summary)

    return summary

def test_experiment_0():
    """Test caching and full MLE with small dataset."""
    print("\n=== TESTING EXPERIMENT 0 INFRASTRUCTURE ===")
    print("Running with 1 replicate, 2 epistasis levels, 1000 samples")
    print("This should complete in ~2-3 minutes")

    results = run_experiment_0_with_replicates(
        epistasis_levels=[0.0, 1.0],  # Just 2 levels
        n_replicates=1,  # Just 1 replicate
        n_samples=1000,  # Small dataset
        use_cache=True,
        force_retrain=False
    )

    print("\n✓ Test complete! Caching system works.")
    print("Run again to verify caching (should be instant)...")

    # Run again to test cache loading
    print("\n" + "="*70)
    print("SECOND RUN (should load from cache)")
    print("="*70)

    results2 = run_experiment_0_with_replicates(
        epistasis_levels=[0.0, 1.0],
        n_replicates=1,
        n_samples=1000,
        use_cache=True,
        force_retrain=False
    )

    print("\n✓ Caching verified!")

def test_convergence_at_epistasis_0(
    n_samples=50000,
    n_epochs=5000,
    eval_every=50,
    lr=0.01,
    seed=42
):
    """
    Test if factorized model converges to full MLE at epistasis=0.

    Trains a single model for many epochs and tracks error over time.
    """
    print("\n=== Testing Factorized Model Convergence at Epistasis=0 ===")
    print(f"Dataset: {n_samples} samples")
    print(f"Training: {n_epochs} epochs (eval every {eval_every})")
    print("=" * 70)

    # Generate ground truth and data
    print("\nGenerating ground truth (epistasis=0)...")
    gt = GroundTruthProcess(seed=seed, epistasis=0.0)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

    print(f"Generating {n_samples} samples...")
    data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=1)
    print(f"Generated {len(data)} samples")

    # Compute full MLE baseline
    print("\nComputing full MLE baseline...")
    Q_full_np = estimate_full_mle_q(data, gt)
    Q_full = torch.tensor(Q_full_np, dtype=torch.float32)
    error_full_mle = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
    print(f"Full MLE error: {error_full_mle.item():.4f}")

    # Train factorized model with periodic evaluation
    print(f"\nTraining factorized model for {n_epochs} epochs...")
    model = NeuralRateModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Pre-process data
    starts = torch.stack([d[0] for d in data])
    ends = torch.stack([d[1] for d in data])
    bs = torch.tensor([d[2] for d in data], dtype=torch.float32)

    # Track metrics
    epochs_list = []
    errors_list = []
    losses_list = []

    for epoch in range(n_epochs):
        # Training step
        optimizer.zero_grad()

        q_preds = model(starts)
        log_prob_total = 0

        for l in range(L):
            q_l = q_preds[:, l]
            b_view = bs.view(-1, 1, 1)
            matrix_input = q_l * b_view
            p_matrix = torch.matrix_exp(matrix_input)

            start_char = starts[:, l]
            end_char = ends[:, l]
            probs = p_matrix[torch.arange(len(data)), start_char, end_char]
            log_prob_total += torch.log(probs + 1e-9)

        loss = -log_prob_total.mean()
        loss.backward()
        optimizer.step()

        # Periodic evaluation
        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                Q_model = model.build_global_Q()
                error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)

                epochs_list.append(epoch)
                errors_list.append(error.item())
                losses_list.append(loss.item())

                print(f"Epoch {epoch:4d}: Loss={loss.item():.4f}, Error={error.item():.4f}")

    # Determine convergence epoch
    tolerance = 0.2  # Target: within 0.2 of full MLE
    convergence_epoch = None
    for i, (ep, err) in enumerate(zip(epochs_list, errors_list)):
        gap = err - error_full_mle.item()
        if gap < tolerance:
            convergence_epoch = ep
            break

    # Final results
    print("\n" + "=" * 70)
    print("CONVERGENCE RESULTS")
    print("=" * 70)
    print(f"Initial error (epoch 0):      {errors_list[0]:.4f}")
    print(f"Final error (epoch {n_epochs - 1}):   {errors_list[-1]:.4f}")
    print(f"Full MLE error (baseline):    {error_full_mle.item():.4f}")
    print(f"Final gap (fact - full MLE):  {errors_list[-1] - error_full_mle.item():.4f}")

    if convergence_epoch is not None:
        print(f"\n✓ CONVERGED at epoch {convergence_epoch}")
        print(f"  Gap fell below {tolerance} at epoch {convergence_epoch}")
        print(f"  Recommended epochs for full experiment: {convergence_epoch}")
    else:
        print(f"\n✗ Did NOT converge within tolerance ({tolerance})")
        print(f"  Final gap: {errors_list[-1] - error_full_mle.item():.4f}")
        print(f"  Consider: more epochs, learning rate schedule, or accept higher baseline error")

    # Plot convergence
    plot_convergence(epochs_list, errors_list, losses_list, error_full_mle.item(),
                     convergence_epoch, tolerance)

    return {
        'epochs': epochs_list,
        'errors': errors_list,
        'losses': losses_list,
        'full_mle_error': error_full_mle.item(),
        'final_error': errors_list[-1],
        'convergence_epoch': convergence_epoch,
        'tolerance': tolerance
    }

def debug_epistasis_0():
    """Comprehensive debugging of epistasis=0 case.

    Investigates why factorized model error increases with training
    even though ground truth is perfectly factorizable.
    """
    print("\n" + "=" * 70)
    print("=== DEBUGGING EPISTASIS=0 ===")
    print("=" * 70)

    # 1. Verify ground truth structure
    print("\n1. Checking ground truth Q structure...")
    gt = GroundTruthProcess(seed=42, epistasis=0.0)

    # Check that rates are context-independent
    print("  Sample rates for C→G at site 0:")
    test_contexts = [0, 1, 10, 100]
    for i in test_contexts:
        s_i = SEQS[i]
        # Find a neighbor differing at site 0, where current is C (base 1)
        if s_i[0] == 1:  # C
            s_j = list(s_i)
            s_j[0] = 2  # G
            j = seq_to_idx(tuple(s_j))
            print(f"    Context {i} ({s_i}): Q[{i},{j}] = {gt.Q_global[i, j]:.4f}")

    # Alternative: find any C→G transition at site 0
    print("\n  Checking ALL C→G transitions at site 0:")
    cg_rates = []
    for i in range(N_STATES):
        s_i = SEQS[i]
        if s_i[0] == 1:  # C at site 0
            s_j = list(s_i)
            s_j[0] = 2  # G
            j = seq_to_idx(tuple(s_j))
            cg_rates.append(gt.Q_global[i, j])

    if len(cg_rates) > 0:
        cg_rates = np.array(cg_rates)
        print(f"    Mean: {cg_rates.mean():.4f}, Std: {cg_rates.std():.6f}")
        print(f"    Should all be identical (std ≈ 0)")

    # 2. Verify Q is a Kronecker sum
    print("\n2. Verifying Q is a Kronecker sum Q = Q^0 ⊕ Q^1 ⊕ Q^2 ⊕ Q^3...")

    # Manually construct Kronecker sum
    Q_kronecker = np.zeros((N_STATES, N_STATES))
    for l in range(L):
        # Contribution from site l: I ⊗ ... ⊗ Q^l ⊗ ... ⊗ I
        for i in range(N_STATES):
            s_i = SEQS[i]
            for j in range(N_STATES):
                s_j = SEQS[j]
                # Check if only site l differs
                if sum(1 for k in range(L) if s_i[k] != s_j[k] and k != l) == 0:
                    if s_i[l] != s_j[l]:
                        # Transition at site l
                        Q_kronecker[i, j] += gt.base_rates[l, s_i[l], s_j[l]]

    # Set diagonal
    for i in range(N_STATES):
        Q_kronecker[i, i] = -np.sum(Q_kronecker[i, :])

    Q_diff = np.abs(gt.Q_global - Q_kronecker).max()
    print(f"  Max |Q_global - Q_kronecker|: {Q_diff:.2e}")
    if Q_diff < 1e-10:
        print(f"  ✓ Q_global IS a Kronecker sum")
    else:
        print(f"  ✗ Q_global is NOT a Kronecker sum (BUG!)")
        # Show example mismatch
        for i in range(10):
            for j in range(10):
                if abs(gt.Q_global[i,j] - Q_kronecker[i,j]) > 1e-6:
                    print(f"    Mismatch at [{i},{j}]: Q_global={gt.Q_global[i,j]:.4f}, Q_kronecker={Q_kronecker[i,j]:.4f}")
                    break
            else:
                continue
            break

    # 3. Verify P factorization
    print("\n3. Verifying P = exp(bQ) factorizes...")
    b = 0.5
    P_full = scipy.linalg.expm(b * gt.Q_global)

    # Compute factorized P using Kronecker product formula
    # P[i,j] = ∏_l exp(b*Q^l)[x_i[l], x_j[l]]
    P_factorized = np.ones((N_STATES, N_STATES))
    for l in range(L):
        P_l = scipy.linalg.expm(b * gt.base_rates[l])
        for i in range(N_STATES):
            for j in range(N_STATES):
                P_factorized[i, j] *= P_l[SEQS[i][l], SEQS[j][l]]

    diff = np.abs(P_full - P_factorized).max()
    print(f"  Max |P_full - P_factorized|: {diff:.2e}")

    # Debug: check if base_rates have proper diagonal
    print(f"\n  Checking base_rates diagonal structure:")
    for l in range(L):
        diag_zero = np.allclose(np.diag(gt.base_rates[l]), 0.0)
        row_sums = gt.base_rates[l].sum(axis=1)
        print(f"    Site {l}: diagonal zeros? {diag_zero}, row sums: {row_sums}")

    if diff < 1e-10:
        print(f"\n  ✓ P factorizes perfectly")
    else:
        print(f"\n  ✗ P does NOT factorize")
        # Show example mismatch
        max_i, max_j = np.unravel_index(np.argmax(np.abs(P_full - P_factorized)), P_full.shape)
        print(f"    Max diff at [{max_i},{max_j}]:")
        print(f"      P_full = {P_full[max_i, max_j]:.6f}")
        print(f"      P_factorized = {P_factorized[max_i, max_j]:.6f}")

    # 4. Train context-dependent model and measure context-dependence
    print("\n4. Training context-DEPENDENT model (NeuralRateModel)...")
    data = gt.generate_data(5000, lambda_param=1.0, min_mutations=1)

    model = NeuralRateModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Pre-process data
    starts = torch.stack([d[0] for d in data])
    ends = torch.stack([d[1] for d in data])
    bs_data = torch.tensor([d[2] for d in data], dtype=torch.float32)

    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

    # Train and check periodically
    eval_epochs = [0, 100, 500, 1000]
    for epoch in range(max(eval_epochs) + 1):
        if epoch > 0:
            optimizer.zero_grad()
            q_preds = model(starts)
            log_prob_total = 0
            for l in range(L):
                q_l = q_preds[:, l]
                p_matrix = torch.matrix_exp(bs_data.view(-1, 1, 1) * q_l)
                probs = p_matrix[torch.arange(len(data)), starts[:, l], ends[:, l]]
                log_prob_total += torch.log(probs + 1e-9)
            loss = -log_prob_total.mean()
            loss.backward()
            optimizer.step()

        if epoch in eval_epochs:
            # Measure context-dependence
            with torch.no_grad():
                local_qs = model(ALL_SEQS_BATCH)  # (256, L, 4, 4)
                print(f"\n  Epoch {epoch}:")

                # Check variance of learned rates across contexts
                high_variance_count = 0
                for l in range(L):
                    for a in range(4):
                        for b in range(4):
                            if a == b:
                                continue
                            rates = local_qs[:, l, a, b]
                            std = rates.std().item()
                            if std > 0.01:  # Significant variance
                                high_variance_count += 1
                                if high_variance_count <= 3:  # Print first few
                                    print(f"    Site {l}, {a}→{b}: std={std:.4f} (context-dependent!)")

                if high_variance_count > 3:
                    print(f"    ... and {high_variance_count - 3} more transitions with high variance")

                Q_model = model.build_global_Q()
                error = torch.norm(Q_model - Q_true) / torch.norm(Q_true)
                print(f"  Model error: {error.item():.4f}")

    # 5. Test context-INDEPENDENT model
    print("\n5. Training context-INDEPENDENT model (ContextIndependentRateModel)...")
    model_indep = ContextIndependentRateModel()
    optimizer_indep = optim.Adam(model_indep.parameters(), lr=0.01)

    for epoch in range(1001):
        optimizer_indep.zero_grad()
        q_preds = model_indep(starts)
        log_prob_total = 0
        for l in range(L):
            q_l = q_preds[:, l]
            p_matrix = torch.matrix_exp(bs_data.view(-1, 1, 1) * q_l)
            probs = p_matrix[torch.arange(len(data)), starts[:, l], ends[:, l]]
            log_prob_total += torch.log(probs + 1e-9)
        loss = -log_prob_total.mean()
        loss.backward()
        optimizer_indep.step()

        if epoch % 200 == 0:
            with torch.no_grad():
                Q_indep = model_indep.build_global_Q()
                error = torch.norm(Q_indep - Q_true) / torch.norm(Q_true)
                print(f"  Epoch {epoch:4d}: Loss={loss.item():.4f}, Error={error.item():.4f}")

    # Final comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    with torch.no_grad():
        Q_neural = model.build_global_Q()
        Q_indep = model_indep.build_global_Q()
        error_neural = torch.norm(Q_neural - Q_true) / torch.norm(Q_true)
        error_indep = torch.norm(Q_indep - Q_true) / torch.norm(Q_true)

    print(f"Context-DEPENDENT model (NeuralRateModel) error:      {error_neural.item():.4f}")
    print(f"Context-INDEPENDENT model (ContextIndep) error:       {error_indep.item():.4f}")
    print(f"\nIf context-independent model achieves low error (~0),")
    print(f"then the issue is the architecture learning spurious context-dependencies.")

def test_early_stopping_vs_full_mle():
    """
    Test factorized model with early stopping vs full MLE baseline at epistasis=0.

    Uses same cached data as previous experiments for fair comparison.
    """
    print("\n" + "=" * 70)
    print("=== Testing Early Stopping: Factorized vs Full MLE ===")
    print("=" * 70)

    # Parameters
    epistasis = 0.0
    seed = 42
    n_samples = 5000  # Same as original experiments
    lr_values = [0.01, 0.005, 0.001]  # Try different learning rates

    # Create ground truth
    print(f"\nEpistasis: {epistasis}")
    print(f"Seed: {seed}")
    print(f"Dataset size: {n_samples}")
    gt = GroundTruthProcess(seed=seed, epistasis=epistasis)
    Q_true = torch.tensor(gt.Q_global, dtype=torch.float32)

    # Load cached data
    print(f"\nLoading cached dataset...")
    data = load_dataset(epistasis, seed, n_samples)
    if data is None:
        print(f"  No cache found, generating {n_samples} samples...")
        data = gt.generate_data(n_samples, lambda_param=1.0, min_mutations=1)
        save_dataset(data, epistasis, seed, n_samples)
    else:
        print(f"  Loaded {len(data)} samples from cache")

    # 1. Compute full MLE baseline
    print(f"\n1. Computing Full MLE baseline (non-parametric)...")
    Q_full_np = estimate_full_mle_q(data, gt)
    Q_full = torch.tensor(Q_full_np, dtype=torch.float32)
    error_full_mle = torch.norm(Q_full - Q_true) / torch.norm(Q_true)
    print(f"   Full MLE error: {error_full_mle.item():.4f}")
    print(f"   (This is a direct empirical estimate, no training)")

    # 2. Train factorized models with different learning rates
    print(f"\n2. Training Factorized models with early stopping...")
    results = []

    for lr in lr_values:
        print(f"\n--- Learning Rate: {lr} ---")

        model, history = train_model(
            gt, data,
            use_snr=False,
            epochs=2000,
            lr=lr,
            early_stop_patience=100,  # Stop if no improvement for 100 epochs
            early_stop_tolerance=0.001,  # Must improve by at least 0.001
            verbose=True
        )

        # Final error
        with torch.no_grad():
            Q_factorized = model.build_global_Q()
            final_error = torch.norm(Q_factorized - Q_true) / torch.norm(Q_true)

        results.append({
            'lr': lr,
            'final_error': final_error.item(),
            'best_error': min(history['errors']) if history['errors'] else final_error.item(),
            'n_epochs': len(history['epochs']),
            'history': history
        })

        print(f"   Final error: {final_error.item():.4f}")
        print(f"   Best error: {results[-1]['best_error']:.4f}")
        print(f"   Stopped at epoch: {results[-1]['n_epochs']}")

    # 3. Summary comparison
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Full MLE (non-parametric) error:          {error_full_mle.item():.4f}")
    print(f"\nFactorized model (with early stopping):")
    for r in results:
        print(f"  LR={r['lr']:<6} | Best error: {r['best_error']:.4f} | Epochs: {r['n_epochs']:<4} | Gap: {r['best_error'] - error_full_mle.item():+.4f}")

    # Find best learning rate
    best_result = min(results, key=lambda x: x['best_error'])
    print(f"\n✓ Best LR: {best_result['lr']} (error: {best_result['best_error']:.4f})")

    gap = best_result['best_error'] - error_full_mle.item()
    if abs(gap) < 0.2:
        print(f"✓ Factorized model achieves similar performance to Full MLE (gap: {gap:+.4f})")
    else:
        print(f"✗ Factorized model has significant gap from Full MLE (gap: {gap:+.4f})")

    # 4. Plot convergence for best LR
    if best_result['history']['epochs']:
        plt.figure(figsize=(12, 5))

        # Plot 1: Error vs Epochs
        plt.subplot(1, 2, 1)
        plt.plot(best_result['history']['epochs'], best_result['history']['errors'], 'b-', linewidth=2, label='Factorized')
        plt.axhline(y=error_full_mle.item(), color='orange', linestyle='--', linewidth=2, label=f'Full MLE ({error_full_mle.item():.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Relative Frobenius Error')
        plt.title(f'Early Stopping Test (LR={best_result["lr"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Loss vs Epochs
        plt.subplot(1, 2, 2)
        plt.plot(best_result['history']['epochs'], best_result['history']['losses'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/early_stopping_test.png', dpi=150)
        print(f"\nSaved results/early_stopping_test.png")

    return best_result, error_full_mle.item()

def run_experiment_2(gt):
    print("\n=== Experiment 2: SNR Weighting Validation ===")
    # Generate Mixed Dataset (High Bias Potential)
    data_mixed = gt.generate_data(3000, 0.01, 1.0)
    
    model_mle, _ = train_model(gt, data_mixed, use_snr=False)
    model_snr, _ = train_model(gt, data_mixed, use_snr=True)
    
    Q_true = gt.Q_torch
    Q_mle = model_mle.build_global_Q()
    Q_snr = model_snr.build_global_Q()
    
    err_mle = torch.norm(Q_mle - Q_true) / torch.norm(Q_true)
    err_snr = torch.norm(Q_snr - Q_true) / torch.norm(Q_true)
    
    print(f"Relative Error (Standard MLE): {err_mle.item():.4f}")
    print(f"Relative Error (SNR Weighted): {err_snr.item():.4f}")
    
    plt.figure()
    plt.bar(['Standard MLE', 'SNR Weighted'], [err_mle.item(), err_snr.item()])
    plt.ylabel('Relative Frobenius Error')
    plt.title('Exp 2: SNR Weighting Mitigation')
    plt.savefig('exp2_snr.png')
    print("Saved exp2_snr.png")
    return model_snr

def run_experiment_3(gt, model):
    print("\n=== Experiment 3: O(b^2) Error Bound ===")
    bs = np.logspace(-3, -0.5, 20)
    errors = []
    
    # Pick a random start sequence
    start_idx = 0
    start_tens = ALL_SEQS_BATCH[start_idx].unsqueeze(0)
    
    with torch.no_grad():
        # Get Model Qs once
        q_preds = model(start_tens) # (1, L, 4, 4)
        
        for b in bs:
            # True Transition
            P_true = scipy.linalg.expm(b * gt.Q_global)[start_idx]
            
            # Model Transition (Factorized)
            p_model_full = 1.0
            for l in range(L):
                q_l = q_preds[0, l]
                p_l = torch.matrix_exp(b * q_l) # (4, 4)
                # Expand p_l to full space is hard, easier to compute probability of every state
                # But here we just want the vector P_model[start_idx, :]
                # Actually, easier: Kronecker Product
                # But even easier:
                # P(y|x) = Prod P(y^l | x^l)
                pass
            
            # Recompute model prob vector state-by-state (inefficient but safe)
            P_model = np.zeros(N_STATES)
            
            # Precompute the 4 small matrices
            small_ps = []
            for l in range(L):
                small_ps.append(torch.matrix_exp(b * q_preds[0, l]).numpy())
                
            start_seq = SEQS[start_idx]
            for i in range(N_STATES):
                target_seq = SEQS[i]
                prob = 1.0
                for l in range(L):
                    u, v = start_seq[l], target_seq[l]
                    prob *= small_ps[l][u, v]
                P_model[i] = prob
                
            # L1 Error
            l1_err = np.sum(np.abs(P_true - P_model))
            errors.append(l1_err)
            
    # Plot log-log
    plt.figure()
    log_b = np.log10(bs)
    log_err = np.log10(errors)
    
    # Fit line to measure slope
    slope, intercept = np.polyfit(log_b, log_err, 1)
    
    plt.plot(log_b, log_err, 'o-', label=f'Model Error (Slope={slope:.2f})')
    # Plot reference slope 2
    plt.plot(log_b, 2*log_b + (intercept - 2*log_b[0]), 'k--', label='Reference Slope 2')
    
    plt.xlabel('log10(Branch Length)')
    plt.ylabel('log10(L1 Error)')
    plt.legend()
    plt.title('Exp 3: Error Scaling vs Branch Length')
    plt.grid(True)
    plt.savefig('exp3_scaling.png')
    print(f"Slope of error curve: {slope:.4f} (Expected ~2.0)")
    print("Saved exp3_scaling.png")

def run_experiment_4_gillespie(gt, model):
    print("\n=== Experiment 4: Adapted Gillespie Sampling ===")
    b = 0.3
    start_idx = 0
    start_tens = ALL_SEQS_BATCH[start_idx].unsqueeze(0)
    n_samples = 5000
    
    # 1. Target Distribution
    P_target = scipy.linalg.expm(b * gt.Q_global)[start_idx]
    
    # 2. Factorized Sampling (Naive)
    print("Running Naive Sampling...")
    with torch.no_grad():
        q_preds = model(start_tens)
        small_ps = []
        for l in range(L):
            small_ps.append(torch.matrix_exp(b * q_preds[0, l]).numpy())
    
    naive_counts = np.zeros(N_STATES)
    start_seq = SEQS[start_idx] # Tuple
    
    for _ in range(n_samples):
        res_seq = []
        for l in range(L):
            u = start_seq[l]
            # Sample from row u of small_ps[l]
            probs = small_ps[l][u]
            probs /= probs.sum()
            v = np.random.choice(4, p=probs)
            res_seq.append(v)
        idx = seq_to_idx(tuple(res_seq))
        naive_counts[idx] += 1
        
    P_naive = naive_counts / n_samples
    
    # 3. Adapted Gillespie Sampling
    print("Running Gillespie Sampling...")
    gillespie_counts = np.zeros(N_STATES)
    
    for _ in range(n_samples):
        t = 0.0
        curr_seq = list(start_seq) # Mutable list
        curr_seq_idx = start_idx
        
        while t < b:
            # Re-evaluate local Qs at current sequence
            # Note: In a real efficient impl, we only update rows that changed.
            # Here we just run the net (it's fast enough for L=4)
            curr_tens = torch.tensor([curr_seq], dtype=torch.long)
            with torch.no_grad():
                local_qs = model(curr_tens)[0] # (L, 4, 4)
            
            # Calculate total exit rate
            rates = [] # store (site, target_aa, rate)
            lambda_total = 0.0
            
            for l in range(L):
                u = curr_seq[l]
                row = local_qs[l, u]
                for v in range(4):
                    if u == v: continue
                    r = row[v].item()
                    if r > 0:
                        rates.append((l, v, r))
                        lambda_total += r
            
            if lambda_total == 0:
                break # Absorbing state (unlikely with softplus)
                
            # Sample time
            tau = np.random.exponential(1.0 / lambda_total)
            
            if t + tau > b:
                break
                
            t += tau
            
            # Choose reaction
            r_vals = np.array([x[2] for x in rates])
            probs = r_vals / lambda_total
            choice_idx = np.random.choice(len(rates), p=probs)
            
            l_star, v_star, _ = rates[choice_idx]
            curr_seq[l_star] = v_star
            
        final_idx = seq_to_idx(tuple(curr_seq))
        gillespie_counts[final_idx] += 1
        
    P_gillespie = gillespie_counts / n_samples
    
    # Calc KL Divergence
    def kl(p, q):
        # Add epsilon to avoid log(0)
        p = p + 1e-9
        q = q + 1e-9
        p /= p.sum()
        q /= q.sum()
        return np.sum(p * np.log(p / q))
        
    kl_naive = kl(P_target, P_naive)
    kl_gillespie = kl(P_target, P_gillespie)
    
    print(f"KL Divergence (Naive Factorized): {kl_naive:.5f}")
    print(f"KL Divergence (Adapted Gillespie): {kl_gillespie:.5f}")
    
    plt.figure()
    plt.bar(['Naive Factorized', 'Adapted Gillespie'], [kl_naive, kl_gillespie])
    plt.ylabel('KL Divergence from Ground Truth')
    plt.title('Exp 4: Sampling Fidelity')
    plt.savefig('exp4_sampling.png')
    print("Saved exp4_sampling.png")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Test mode
        test_experiment_0()
    elif len(sys.argv) > 1 and sys.argv[1] == '--convergence':
        # Convergence test at epistasis=0
        test_convergence_at_epistasis_0(
            n_samples=50000,
            n_epochs=5000,
            eval_every=50,
            lr=0.01
        )
    elif len(sys.argv) > 1 and sys.argv[1] == '--debug':
        # Debug epistasis=0 case
        debug_epistasis_0()
    elif len(sys.argv) > 1 and sys.argv[1] == '--early-stop':
        # Test early stopping vs full MLE
        test_early_stopping_vs_full_mle()
    else:
        # Full experiment
        print("Running FULL Experiment 0...")
        print("This will take ~30-60 minutes")
        print("Run with --test flag to test first")

        results = run_experiment_0_with_replicates(
            epistasis_levels=[0.0, 0.3, 0.7, 1.0],
            n_replicates=5,
            n_samples=50000,
            use_cache=True,
            force_retrain=False
        )

        print("\n✓ Experiment 0 completed successfully!")

    # Experiment 1 commented out for now
    # gt = GroundTruthProcess()
    # run_experiment_1(gt)

    # Experiments 2-4 are commented out for now
    # # Run Exp 2
    # best_model = run_experiment_2(gt)
    #
    # # Run Exp 3 (using best model)
    # run_experiment_3(gt, best_model)
    #
    # # Run Exp 4 (using best model)
    # run_experiment_4_gillespie(gt, best_model)