import matplotlib.pyplot as plt
import polars as pl
import missingno as mn

def plot_null(df, group, max_samples = 1000000):
    X = df.select(pl.col(f'^.*{group}$')).collect()
    if len(X) > max_samples:
        X = X.sample(max_samples, seed = 0)
    n = max(X.shape[1], 8)
    X = X.to_pandas(use_pyarrow_extension_array = True)
    fig, axes = plt.subplots(2, figsize = (n, int(1.6 * n)))
    mn.matrix(X, ax = axes[0], sparkline = False)
    mn.heatmap(X, ax = axes[1])
    fig.tight_layout()