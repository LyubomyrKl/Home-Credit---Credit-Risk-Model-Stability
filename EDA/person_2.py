from utils.scan_and_parse_date import scan_data_and_parse_dates

import polars as pl
import numpy as np

train_person_2 = scan_data_and_parse_dates(
    '../home-credit-credit-risk-model-stability/parquet_files/train/train_person_2.parquet').lazy().collect()


def get_pmts_end_date(list_of_series):
    return list_of_series[0].max()


def get_mask(list_of_series):
    count_unique = list_of_series[0].unique()
    return list_of_series[0].mode()[0] if len(count_unique) > 1 else list_of_series[0][0]


train_person_2_to_1 = train_person_2[:20].group_by(['case_id', 'num_group1']).agg(
    pl.map_groups(exprs=["addres_district_368M"], function=get_mask).alias("addres_district"),
    pl.map_groups(exprs=["addres_zip_823M"], function=get_pmts_end_date).alias("addres_zip"),
    pl.map_groups(exprs=["conts_role_79M"], function=get_mask).alias("conts_role"),
    pl.map_groups(exprs=["empls_economicalst_849M"], function=get_mask).alias("empls_economicalst"),
    pl.map_groups(exprs=["empls_employer_name_740M"], function=get_mask).alias("empls_employer_name"),
)

train_person_2_to_1.write_parquet('person_2.parquet')

unique_id = train_person_2.filter(train_person_2['relatedpersons_role_762T'] != np.nan)['case_id'].unique()

