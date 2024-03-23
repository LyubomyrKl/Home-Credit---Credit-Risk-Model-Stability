import pandas as pd
import numpy as np

def inspect_columns(df):
    df = df.to_pandas()

    result = pd.DataFrame({
        'unique': df.nunique() == len(df),
        'cardinality': df.nunique(),
        'with_null': df.isna().any(),
        'null_pct': round((df.isnull().sum() / len(df)) * 100, 2),
        '1st_row': df.iloc[0],
        'random_row': df.iloc[np.random.randint(low=0, high=len(df))],
        'last_row': df.iloc[-1],
        'dtype': df.dtypes
    })

    return result


def custom_format(x):
    if x % 1 == 0:
        return int(x)
    else:
        return '%.2f' % x


# Set display options
pd.set_option('display.float_format', custom_format)