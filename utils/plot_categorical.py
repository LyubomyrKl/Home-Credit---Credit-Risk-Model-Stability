import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

def plot_categorical(df, group, max_categories = 30, max_samples = 1000000):
    X = df.select(pl.col(f'^.*{group}$')).collect()
    if len(X) > max_samples:
        X = X.sample(max_samples, seed = 0)
    n = X.shape[1]
    fig, axes = plt.subplots(n, 2, figsize = (16, n * 4), squeeze = False)
    for i, c in enumerate(X.iter_columns()):
        x = c.to_pandas()
        s = x.value_counts()
        if len(s) > max_categories:
            other_count = s[max_categories:].sum()
            s = s[:max_categories]
            s['Other'] = other_count
        sns.barplot(x = s.index, y = s, ax = axes[i, 0])
        axes[i, 1].pie(s, labels = s.index)
    fig.tight_layout()
