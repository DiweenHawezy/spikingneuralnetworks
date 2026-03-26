import numpy as np

def transfer_entropy_debug(x, y, lag=1, n_bins=10):
    """
    Compute Transfer Entropy with debug output.
    """
    n = len(x) - lag
    
    if n <= 0:
        return 0.0
    
    # Discretize the time series using actual data ranges
    x_disc = np.digitize(x[:n], np.linspace(x.min(), x.max(), n_bins+1)) - 1
    y_past = np.digitize(y[:-lag], np.linspace(y.min(), y.max(), n_bins+1)) - 1
    y_future = np.digitize(y[lag:], np.linspace(y.min(), y.max(), n_bins+1)) - 1
    
    # Ensure we have valid bin indices (0 to n_bins-1)
    x_disc = np.clip(x_disc, 0, n_bins-1)
    y_past = np.clip(y_past, 0, n_bins-1)
    y_future = np.clip(y_future, 0, n_bins-1)
    
    print(f"x_disc range: [{x_disc.min()}, {x_disc.max()}]")
    print(f"y_past range: [{y_past.min()}, {y_past.max()}]")
    print(f"y_future range: [{y_future.min()}, {y_future.max()}]")
    
    # Compute joint probabilities using histograms
    # P(X_past, Y_past, Y_future)
    hist_3d = np.zeros((n_bins, n_bins, n_bins))
    for i in range(n):
        hist_3d[x_disc[i], y_past[i], y_future[i]] += 1
    p_xyp_yf = hist_3d / np.sum(hist_3d)
    
    # P(Y_past, Y_future)
    hist_2d = np.zeros((n_bins, n_bins))
    for i in range(n):
        hist_2d[y_past[i], y_future[i]] += 1
    p_yyp_yf = hist_2d / np.sum(hist_2d)
    
    # P(X_past, Y_past)
    hist_xy = np.zeros((n_bins, n_bins))
    for i in range(n):
        hist_xy[x_disc[i], y_past[i]] += 1
    p_xyp = hist_xy / np.sum(hist_xy)
    
    # P(Y_future)
    hist_yf = np.zeros(n_bins)
    for i in range(n):
        hist_yf[y_future[i]] += 1
    p_yf = hist_yf / np.sum(hist_yf)
    
    # P(Y_past)
    hist_yp = np.zeros(n_bins)
    for i in range(n):
        hist_yp[y_past[i]] += 1
    p_yp = hist_yp / np.sum(hist_yp)
    
    print(f"p_yp: {p_yp}")
    print(f"p_yf: {p_yf}")
    print(f"p_xyp sum: {np.sum(p_xyp)}")
    print(f"p_xyp_yf sum: {np.sum(p_xyp_yf)}")
    
    # Compute TE using the formula:
    # TE = sum P(X,Y_past,Y_future) * log(P(X,Y_past,Y_future) * P(Y_past) / (P(X,Y_past) * P(Y_future)))
    te = 0.0
    nonzero_count = 0
    for i in range(n_bins):  # x_disc
        for j in range(n_bins):  # y_past
            for k in range(n_bins):  # y_future
                if p_xyp_yf[i, j, k] > 0:
                    term = p_xyp_yf[i, j, k] * p_yp[j] / (p_xyp[i, j] * p_yf[k])
                    if term > 0:
                        te += p_xyp_yf[i, j, k] * np.log2(term)
                        nonzero_count += 1
    
    print(f"Non-zero terms: {nonzero_count}")
    print(f"TE: {te}")
    
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

print("\nTransfer Entropy from x to y (Lag 1):")
transfer_entropy_debug(x_norm[:n-1], y_norm, lag=1, n_bins=8)
