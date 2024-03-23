import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

def plot_corr(df, group, max_samples = 1000000):
    X = df.select(pl.col(f'^.*{group}$')).collect()
    if len(X) > max_samples:
        X = X.sample(max_samples, seed = 0)
    corr = X.to_pandas(use_pyarrow_extension_array = True).corr()
    w = len(corr)
    h = int(w * 0.8)
    _, ax = plt.subplots(figsize = (w, h))
    sns.heatmap(corr, ax = ax, vmin = -1, vmax = 1, annot = True, cmap = 'coolwarm')