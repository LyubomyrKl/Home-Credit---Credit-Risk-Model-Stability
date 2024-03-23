import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

def plot_numerical(df, group, max_samples = 1000000):
    X = df.select(pl.col(f'^.*{group}$')).collect()
    if len(X) > max_samples:
        X = X.sample(max_samples, seed = 0)
    n = X.shape[1]
    fig, axes = plt.subplots(n, 2, figsize = (16, n * 4), squeeze = False)
    for i, c in enumerate(X.iter_columns()):
        x = c.to_pandas()
        sns.histplot(x = x, bins = 50, kde = True, ax = axes[i, 0])
        sns.boxplot(y = x, ax = axes[i, 1])
    fig.tight_layout()