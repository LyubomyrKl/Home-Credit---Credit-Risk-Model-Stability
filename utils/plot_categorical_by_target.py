import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl



def plot_categorical_by_target(df_base, df, group, max_samples=1000000):
    df_base = df_base.lazy()
    X = df_base.join(
        df, on='case_id', how='left',
    ).select('target', pl.col(f'^.*{group}$')).collect()

    if len(X) > max_samples:
        X = X.sample(max_samples, seed=0)

    y = X['target'].to_pandas()
    X = X.drop('target')

    n = X.shape[1]
    fig, axes = plt.subplots(n, 2, figsize=(16, n * 4), squeeze=False)

    for i, c in enumerate(X.iter_columns()):
        x = c.to_pandas()
        sns.violinplot(x=y, y=x, orient='v', order=[0, 1], ax=axes[i, 0])
        # sns.stripplot(x=y, y=x, orient='v', order=[0, 1], ax=axes[i, 1], marker='.')

    fig.tight_layout()
