from utils.scan_and_parse_date import scan_data_and_parse_dates
import polars as pl

train_tax_registry_a_1 = scan_data_and_parse_dates(
    '../home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_a_1.parquet').lazy().collect().rename(
    {
        'amount_4527230A': 'amount',
        'name_4527232M': 'name',
        'recorddate_4527225D': 'record_date'
    }).select(['case_id', 'num_group1', 'name', 'amount', 'record_date'])

train_tax_registry_b_1 = scan_data_and_parse_dates(
    '../home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_b_1.parquet').lazy().collect().rename(
    {
        'amount_4917619A': 'amount',
        'name_4917606M': 'name',
        'deductiondate_4917603D': 'record_date'
    }).select(['case_id', 'num_group1', 'name', 'amount', 'record_date'])

train_tax_registry_c_1 = scan_data_and_parse_dates(
    '../home-credit-credit-risk-model-stability/parquet_files/train/train_tax_registry_c_1.parquet').lazy().collect().rename(
    {
        'pmtamount_36A': 'amount',
        'employername_160M': 'name',
        'processingdate_168D': 'record_date'
    }).select(['case_id', 'num_group1', 'name', 'amount', 'record_date'])

unique_a = train_tax_registry_a_1['case_id'].unique()
unique_b = train_tax_registry_b_1['case_id'].unique()
unique_c = train_tax_registry_c_1['case_id'].unique()

commonUniqueC = unique_a.filter(unique_a.is_in(unique_c))
commonUniqueB = unique_a.filter(unique_a.is_in(unique_b))

reg_a = train_tax_registry_a_1.filter(~pl.col('case_id').is_in(commonUniqueC) & ~pl.col('case_id').is_in(commonUniqueB))

print(len(unique_b) + len(unique_c) + len(reg_a['case_id'].unique()))

taxes = pl.concat([reg_a, train_tax_registry_b_1, train_tax_registry_c_1])
taxes.write_parquet('taxes.parquet')
