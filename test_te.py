import numpy as np

def transfer_entropy_simple(x, y, lag=1, n_bins=10):
    """
    Compute Transfer Entropy using binned mutual information.
    """
    n = len(x) - lag
    
    if n <= 0:
        return 0.0
    
    # Discretize the time series
    x_disc = np.digitize(x[:n], np.linspace(0, n+1, n_bins+1)) - 1
    y_past = np.digitize(y[:-lag], np.linspace(0, n+1, n_bins+1)) - 1
    y_future = np.digitize(y[lag:], np.linspace(0, n+1, n_bins+1)) - 1
    
    # Ensure we have valid bin indices
    x_disc = np.clip(x_disc, 0, n_bins-1)
    y_past = np.clip(y_past, 0, n_bins-1)
    y_future = np.clip(y_future, 0, n_bins-1)
    
    # P(Y_past, Y_future)
    p_y_past_y_future = np.zeros((n_bins, n_bins))
    for i in range(n):
        p_y_past_y_future[y_past[i], y_future[i]] += 1
    p_y_past_y_future = p_y_past_y_future / np.sum(p_y_past_y_future)
    
    # P(Y_past, X_past, Y_future)
    p_x_y_past_y_future = np.zeros((n_bins, n_bins, n_bins))
    for i in range(n):
        p_x_y_past_y_future[x_disc[i], y_past[i], y_future[i]] += 1
    p_x_y_past_y_future = p_x_y_past_y_future / np.sum(p_x_y_past_y_future)
    
    def conditional_entropy(p_joint, axes):
        """Compute conditional entropy H(B|A) where p_joint is P(A,B)."""
        if isinstance(axes, list):
            marginal = np.sum(p_joint, axis=tuple(axes), keepdims=True)
        else:
            marginal = np.sum(p_joint, axis=axes, keepdims=True)
        marginal = np.where(marginal > 0, marginal, 1)
        p_cond = p_joint / marginal
        p_cond = np.where(p_cond > 0, p_cond, 1)
        h = -np.sum(p_joint * np.log2(p_cond))
        return h
    
    h_y_future_given_y_past = conditional_entropy(p_y_past_y_future, 0)
    h_y_future_given_x_y_past = conditional_entropy(p_x_y_past_y_future, (0, 1))
    
    te = h_y_future_given_y_past - h_y_future_given_x_y_past
    
    return max(0, te)

# Test with simple correlated data
np.random.seed(42)
n = 1000

# Create correlated time series
x = np.random.randn(n)
y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)  # y depends on past x

# Normalize
x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

print(f"x range: [{x_norm.min():.4f}, {x_norm.max():.4f}]")
print(f"y range: [{y_norm.min():.4f}, {y_norm.max():.4f}]")

print("\nTransfer Entropy from x to y:")
for lag in range(1, 6):
    if len(x_norm) > lag:
        te = transfer_entropy_simple(x_norm[:n-1], y_norm, lag=lag, n_bins=8)
        print(f"  Lag {lag}: TE = {te:.6f}")

# Test with independent data
print("\nTransfer Entropy with independent data:")
z = np.random.randn(n-1)
print(f"  Lag 1: TE = {transfer_entropy_simple(x_norm[:n-1], z, lag=1, n_bins=8):.6f}")
