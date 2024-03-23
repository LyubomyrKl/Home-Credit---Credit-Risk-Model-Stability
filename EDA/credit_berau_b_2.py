from utils.scan_and_parse_date import scan_data_and_parse_dates
import polars as pl
import numpy as np

credit_bureau_b_2 = scan_data_and_parse_dates(
    '/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_b_2.parquet').lazy().collect()


def get_pmts_start_date(list_of_series):
    return list_of_series[0].min()


def get_pmts_end_date(list_of_series):
    return list_of_series[0].max()


def get_pmts_mean(list_of_series):
    values = np.trim_zeros(np.nan_to_num(np.array(list_of_series[0])))
    return np.mean(values) if len(values) > 0 else 0.


def get_pmts_overdue_percent(list_of_series):
    sample = np.nan_to_num(np.array(list_of_series[0]))

    non_trim_length = len(sample)
    trim_length = len(np.trim_zeros(sample))

    return trim_length / non_trim_length


bereau_b_2_to_1 = credit_bureau_b_2.group_by(['case_id', 'num_group1']).agg(
    pl.map_groups(exprs=["pmts_date_1107D"], function=get_pmts_start_date).alias("pmts_start_date"),
    pl.map_groups(exprs=["pmts_date_1107D"], function=get_pmts_end_date).alias("pmts_end_date"),
    pl.map_groups(exprs=["pmts_dpdvalue_108P"], function=get_pmts_mean).alias("pmts_dpdvalue_mean"),
    pl.map_groups(exprs=["pmts_pmtsoverdue_635A"], function=get_pmts_mean).alias("pmts_pmtsoverdue_mean"),
    pl.map_groups(exprs=["pmts_pmtsoverdue_635A"], function=get_pmts_overdue_percent).alias("pmts_overdue_percent"),
)

nm_unique = credit_bureau_b_2['case_id'].unique()
uniqewjkds = bereau_b_2_to_1['case_id'].unique()
print(bereau_b_2_to_1.filter(pl.col('case_id') == 11196))
bereau_b_2_to_1.write_parquet('bereau_b_2.parquet')
