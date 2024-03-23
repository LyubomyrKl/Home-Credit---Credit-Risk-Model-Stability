import polars as pl

def scan_data_and_parse_dates(parquet_file):
    return pl.scan_parquet(parquet_file).with_columns(
        pl.col(r'^.*D$').str.to_datetime(),
    )