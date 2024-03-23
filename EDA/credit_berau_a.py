from utils.scan_and_parse_date import scan_data_and_parse_dates
import polars as pl
import numpy as np
import datetime
from scipy import stats

df_train_credit_bureau_a_1 = scan_data_and_parse_dates(
    '/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_1_*.parquet')
credit_bureau_a_1 = df_train_credit_bureau_a_1.lazy().collect()

credit_bureau_a_2 = scan_data_and_parse_dates(
    '/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_*.parquet').lazy().collect()
cread_var = scan_data_and_parse_dates('/kaggle/working/bereau_a_2(1)_1.parquet').lazy().collect()


def find_ceil_sum(list_of_series):
    return np.ceil(np.nan_to_num(np.array(list_of_series[0]))).sum().astype('int')


def find_average(list_of_series):
    non_zero_values = np.trim_zeros(np.nan_to_num(np.array(list_of_series[0])))
    length = len(non_zero_values)
    return non_zero_values.sum() / length if length > 0 else 0.


def pmts_overdue_active_day_percent(list_of_series):
    list_array = np.ceil(np.nan_to_num(np.array(list_of_series[0]))).astype('int')
    list_len = len(list_array)

    unique, counts = np.unique(list_array, return_counts=True)
    dictOfUniqueValue = dict(zip(unique, counts))
    is_null_in_dict = 0 in dictOfUniqueValue

    return (1 - dictOfUniqueValue[0] / list_len) if is_null_in_dict else 1.


def pmts_overdue_weighted_average(list_of_series):
    list_array = np.ceil(np.nan_to_num(np.array(list_of_series[0])))
    list_len = len(list_array)

    return list_array.sum() / list_len


def getMode(list_of_series):
    mode = stats.mode(np.array(list_of_series[0])).mode
    return int(mode) if not np.isnan(mode) else 0


def getStartDate(list_of_series):
    years = np.nan_to_num(np.array(list_of_series[0])).astype('int')
    if years.sum() == 0:
        return 0

    year = years[years != 0].min()

    months = np.nan_to_num(np.array(list_of_series[1])).astype('int')
    month = np.trim_zeros(months)[0]

    if month:
        return datetime.datetime(year, month, 1)
    else:
        return datetime.datetime(year, 1, 1)


def getEndDate(list_of_series):
    years = np.nan_to_num(np.array(list_of_series[0])).astype('int')
    if years.sum() == 0:
        return 0

    year = years.max()

    # convert fill nan with zeros and convert it from float to int
    months = np.nan_to_num(np.array(list_of_series[1])).astype('int')

    # get first non-zero value from the end
    month = np.trim_zeros(months[::-1])[0]

    if month:
        return datetime.datetime(year, month, 1)
    else:
        return datetime.datetime(year, 1, 1)


def select_mask(list_of_series):
    rawArray = np.array(list_of_series)
    filtered_arr = rawArray[rawArray != "a55475b1"]
    return filtered_arr[0] if len(filtered_arr) > 0 else 'a55475b1'


already_done = 0
for i in range(11 - already_done):
    # Read and parse data
    credit_bureau_a_2 = scan_data_and_parse_dates(
        f'/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train/train_credit_bureau_a_2_{i + already_done}.parquet').lazy().collect()

    # Group by 'case_id' and 'num_group1' and perform aggregations
    result_df = credit_bureau_a_2.group_by(['case_id', 'num_group1']).agg(
        # Active
        pl.map_groups(exprs=["pmts_year_1139T"], function=getMode).alias("pmts_year_active_mode"),
        pl.map_groups(exprs=["pmts_year_1139T", "pmts_month_158T"], function=getStartDate).alias(
            "pmts_year_active_start"),
        pl.map_groups(exprs=["pmts_year_1139T", "pmts_month_158T"], function=getEndDate).alias(
            "pmts_year_active_finish"),

        pl.map_groups(exprs=["pmts_overdue_1140A"], function=find_ceil_sum).alias("pmts_overdue_active_sum"),
        pl.map_groups(exprs=["pmts_overdue_1140A"], function=pmts_overdue_active_day_percent).alias(
            "pmts_overdue_active_day_percent"),
        pl.map_groups(exprs=["pmts_overdue_1140A"], function=pmts_overdue_weighted_average).alias(
            "pmts_overdue_active_weighted_average"),

        pl.map_groups(exprs=["pmts_dpd_1073P"], function=find_ceil_sum).alias("pmts_dpd_active"),

        pl.map_groups(exprs=["collater_valueofguarantee_1124L"], function=find_average).alias("collater_value_active"),

        pl.map_groups(exprs=["collater_typofvalofguarant_298M"], function=select_mask).alias(
            "collater_typofvalofguarant_active"),

        pl.map_groups(exprs=["collaterals_typeofguarante_669M"], function=select_mask).alias(
            "collaterals_typeofguarante_active"),

        pl.map_groups(exprs=["subjectroles_name_838M"], function=select_mask).alias("subjectroles_name_active"),

        # Closed
        pl.map_groups(exprs=["pmts_year_507T"], function=getMode).alias("pmts_year_closed_mode"),
        pl.map_groups(exprs=["pmts_year_507T", "pmts_month_706T"], function=getStartDate).alias(
            "pmts_year_closed_start"),
        pl.map_groups(exprs=["pmts_year_507T", "pmts_month_706T"], function=getEndDate).alias(
            "pmts_year_closed_finish"),

        pl.map_groups(exprs=["pmts_overdue_1152A"], function=find_ceil_sum).alias("pmts_overdue_closed_sum"),
        pl.map_groups(exprs=["pmts_overdue_1152A"], function=pmts_overdue_active_day_percent).alias(
            "pmts_overdue_closed_day_percent"),
        pl.map_groups(exprs=["pmts_overdue_1152A"], function=pmts_overdue_weighted_average).alias(
            "pmts_overdue_closed_weighted_average"),

        pl.map_groups(exprs=["pmts_dpd_303P"], function=find_ceil_sum).alias("pmts_dpd_closed"),

        pl.map_groups(exprs=["collater_valueofguarantee_876L"], function=find_average).alias("collater_value_closed"),

        pl.map_groups(exprs=["collater_typofvalofguarant_407M"], function=select_mask).alias(
            "collater_typofvalofguarant_closed"),

        pl.map_groups(exprs=["collaterals_typeofguarante_359M"], function=select_mask).alias(
            "collaterals_typeofguarante_closed"),

        pl.map_groups(exprs=["subjectroles_name_541M"], function=select_mask).alias("subjectroles_name_closed")
    )

    # Write the result to parquet file
    result_df.write_parquet(f'bereau_a_2_{i + already_done}.parquet')

    # Print progress
    print(f'{i + already_done}/10')

credit_bureau_a_2 = scan_data_and_parse_dates('/kaggle/working/bereau_a_2_*.parquet').lazy().collect()
credit_bureau_a_2.write_parquet('bereau_a_2.parquet')
