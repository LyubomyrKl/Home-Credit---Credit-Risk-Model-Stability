import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb

import matplotlib.pyplot as plt
from glob import glob

from EDA.static import static_internal_relevants
from paths import TRAIN_DIR, TEST_DIR, ROOT

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))

        return df

    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())

        df = df.drop("date_decision", "MONTH")

        return df

    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]

        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        return expr_max

    @staticmethod
    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)

        return exprs


def feature_eng(df_base, depth_0):
    df_base = (
        df_base
        .with_columns(
            month_decision=pl.col("date_decision").dt.month(),
            weekday_decision=pl.col("date_decision").dt.weekday(),
        )
    )

    for i, df in enumerate(depth_0):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")

    df_base = df_base.pipe(Pipeline.handle_dates)

    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()

    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)

    df_data[cat_cols] = df_data[cat_cols].astype("category")

    return df_data, cat_cols


def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)

    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))

    return df


def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))

    df = pl.concat(chunks, how="vertical_relaxed")
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))

    return df


def check_feature_importances(fitted_models, features_names, PLOT_TOP_N=50, figsize=(10, 12)):
    # Initialize an empty list to store feature importances
    fitted_importances = []

    # Iterate over each fitted model
    for model in fitted_models:
        # Collect feature importances from the model
        importances = model.feature_importances_
        # Append the feature importances to the list
        fitted_importances.append(importances)

    def plot_chinese_importance(importances, features_names, figsize):
        importance_df = pd.DataFrame(data=importances, columns=features_names)
        sorted_indices = importance_df.median(axis=0).sort_values(ascending=False).index
        sorted_importance_df = importance_df.loc[:, sorted_indices]
        plot_cols = sorted_importance_df.columns[:PLOT_TOP_N]
        _, ax = plt.subplots(figsize=figsize)
        ax.grid()
        ax.set_xscale('log')
        ax.set_ylabel('Feature')
        ax.set_xlabel('Importance')
        sns.boxplot(data=sorted_importance_df[plot_cols],
                    orient='h',
                    ax=ax)
        plt.savefig('importances', bbox_inches='tight')
        plt.show()

    plot_chinese_importance(fitted_importances, features_names, figsize)


data_train_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
}

df_train = feature_eng(**data_train_store)
print("train data shape:\t", df_train.shape)

data_test_store = {
    "df_base": read_file(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        read_file(TEST_DIR / "test_static_cb_0.parquet"),
        read_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
}

df_test = feature_eng(**data_test_store)
print("test data shape:\t", df_test.shape)

df_train_copy = df_train.clone()

for column in static_internal_relevants['green']:
    df_train_copy = df_train_copy.with_columns((pl.col(column) / pl.col('credamount_770A')).alias(f'new_{column}'))

df_train_copy = df_train_copy.pipe(Pipeline.filter_cols)

df_test_copy = df_test.clone()

for column in static_internal_relevants['green']:
    df_test_copy = df_test_copy.with_columns((pl.col(column) / pl.col('credamount_770A')).alias(f'new_{column}'))

df_test_copy = df_test_copy.select([col for col in df_train_copy.columns if col != "target"])

print("train data shape:\t", df_train_copy.shape)
print("test data shape:\t", df_test_copy.shape)

print("train data columns:\t", df_train_copy.columns)

df_train, cat_cols = to_pandas(df_train_copy)
df_test, cat_cols = to_pandas(df_test_copy, cat_cols)

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "device": "gpu",
}


def train_voting_model(df_train, params, show_importances=False):
    X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    y = df_train["target"]
    weeks = df_train["WEEK_NUM"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True)

    fitted_models = []

    for idx_train, idx_valid in cv.split(X, y, groups=weeks):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)]
        )

        fitted_models.append(model)

    if show_importances: check_feature_importances(fitted_models, X.columns)

    return VotingModel(fitted_models)


model = train_voting_model(df_train, params)

X_test = df_test.drop(columns=["WEEK_NUM"])
X_test = X_test.set_index("case_id")

y_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = y_pred
print("Check null: ", df_subm["score"].isnull().any())

df_subm.head()
