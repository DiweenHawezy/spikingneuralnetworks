import numpy as np

def transfer_entropy_simple(x, y, lag=1, n_bins=10):
    """
    Compute Transfer Entropy using binned mutual information.
    
    TE(X->Y) = I(Y_future; X_past | Y_past)
             = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    
    Using the formula:
    TE = sum P(X_past, Y_past, Y_future) * log2(
           P(X_past, Y_past, Y_future) * P(Y_past) /
           (P(X_past, Y_past) * P(Y_future))
         )
    """
    n = len(x) - lag
    
    if n <= 0:
        return 0.0
    
    # Discretize the time series
    x_disc = np.digitize(x[:n], np.linspace(x.min(), x.max(), n_bins+1)) - 1
    y_past = np.digitize(y[:-lag], np.linspace(y.min(), y.max(), n_bins+1)) - 1
    y_future = np.digitize(y[lag:], np.linspace(y.min(), y.max(), n_bins+1)) - 1
    
    # Clip to valid bin indices
    x_disc = np.clip(x_disc, 0, n_bins-1)
    y_past = np.clip(y_past, 0, n_bins-1)
    y_future = np.clip(y_future, 0, n_bins-1)
    
    # Compute joint probabilities using np.histogram2d and np.histogram
    # P(X_past, Y_past, Y_future)
    hist_3d, bin_edges = np.histogramdd(
        np.column_stack([x_disc, y_past, y_future]),
        bins=[n_bins, n_bins, n_bins]
    )
    p_3d = hist_3d / np.sum(hist_3d)
    
    # P(Y_past, Y_future) - marginalize over X
    hist_2d, y_edges_1, y_edges_2 = np.histogram2d(y_past, y_future, bins=[n_bins, n_bins])
    p_2d = hist_2d / np.sum(hist_2d)
    
    # P(X_past, Y_past) - marginalize over Y_future
    hist_xy, x_edges, y_edges_xy = np.histogram2d(x_disc, y_past, bins=[n_bins, n_bins])
    p_xy = hist_xy / np.sum(hist_xy)
    
    # P(Y_future)
    hist_yf, _ = np.histogram(y_future, bins=n_bins, range=(0, n_bins))
    p_yf = hist_yf / np.sum(hist_yf)
    
    # P(Y_past)
    hist_yp, _ = np.histogram(y_past, bins=n_bins, range=(0, n_bins))
    p_yp = hist_yp / np.sum(hist_yp)
    
    # Compute TE: I(X_past; Y_future | Y_past)
    # = sum_{x,y_past,y_future} P(x,y_past,y_future) * log2(
    #     P(x,y_past,y_future) * P(y_past) / (P(x,y_past) * P(y_future))
    #   )
    te = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                if p_3d[i, j, k] > 0 and p_xy[i, j] > 0 and p_yf[k] > 0:
                    ratio = (p_3d[i, j, k] * p_yp[j]) / (p_xy[i, j] * p_yf[k])
                    if ratio > 0:
                        te += p_3d[i, j, k] * np.log2(ratio)
    
    return max(0.0, te)

# Test with simple correlated data
np.random.seed(42)
n = 1000

# Create correlated time series where y depends on past x
x = np.random.randn(n)
y = 0.8 * x[:-1] + 0.2 * np.random.randn(n-1)  # y depends on past x

# Normalize
x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

print(f"x range: [{x_norm.min():.4f}, {x_norm.max():.4f}]")
print(f"y range: [{y_norm.min():.4f}, {y_norm.max():.4f}]")

print("\nTransfer Entropy from x to y (correlated data):")
for lag in range(1, 6):
    if len(x_norm) > lag:
        te = transfer_entropy_simple(x_norm[:n-1], y_norm, lag=lag, n_bins=8)
        print(f"  Lag {lag}: TE = {te:.6f}")

# Test with independent data
print("\nTransfer Entropy with independent data:")
z = np.random.randn(n-1)
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)
for lag in range(1, 6):
    te = transfer_entropy_simple(x_norm[:n-1], z_norm, lag=lag, n_bins=8)
    print(f"  Lag {lag}: TE = {te:.6f}")
