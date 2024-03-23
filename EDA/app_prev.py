from utils.scan_and_parse_date import scan_data_and_parse_dates
import polars as pl



df_train_applprev_1 = scan_data_and_parse_dates('/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_*.parquet')
df_train_applprev_2 = scan_data_and_parse_dates('/kaggle/input/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_2.parquet')

app_prev1 = df_train_applprev_1.lazy().collect()
app_prev2 = df_train_applprev_2.lazy().collect()

import pickle
encoded_decoded  = {
     '0': 'null',
     '1': 'PHONE',
     '2': 'ALTERNATIVE_PHONE',
     '3': 'PRIMARY_EMAIL',
     '4': 'SECONDARY_MOBILE',
     '5': 'EMPLOYMENT_PHONE',
     '6': 'WHATSAPP',
     '7': 'PRIMARY_MOBILE',
     '8': 'SKYPE',
     '9': 'HOME_PHONE'
}

decoded_encoded = {v: k for k, v in encoded_decoded.items()}
phone_encoder_pickle_path = 'phone_decoder.pickle'
with open(phone_encoder_pickle_path, 'wb') as file:
    # Dump the dictionary into the file
    pickle.dump(encoded_decoded, file)

with open(phone_encoder_pickle_path, 'rb') as file:
    # Load the dictionary from the file
    loaded_dict = pickle.load(file)


def extract_conts_type_509L(list_of_series):
    array = []
    for item in list_of_series[0]:
        array.append(decoded_encoded[item if item else "null"])
    return ",".join(array)


def extract_cacccardblochreas_147M(list_of_series):
    for value in list_of_series[0]:
        if value is not None:
            return value
    return None


def extract_credacc_cards_status_52L(list_of_series):
    tokens = []
    for value in list_of_series[0]:
        if value is not None:
             tokens.append(value)
    return ','.join(tokens) if len(tokens) > 0 else None


result_df = app_prev2.group_by(['case_id', 'num_group1']).agg(
    pl.map_groups(
            exprs=["conts_type_509L"],
            function=extract_conts_type_509L
        ).alias("conts_type_509L"),
    pl.map_groups(
            exprs=["cacccardblochreas_147M"],
            function=extract_cacccardblochreas_147M
        ).alias("cacccardblochreas_147M"),
    pl.map_groups(
            exprs=["credacc_cards_status_52L"],
            function=extract_credacc_cards_status_52L,
        ).alias("credacc_cards_status_52L")
)



result_df = result_df.with_columns(
    result_df['credacc_cards_status_52L'].map_elements(lambda value: value[0] if isinstance(value, list) else value, skip_nulls=False),
    result_df['cacccardblochreas_147M'].map_elements(lambda value: value[0] if isinstance(value, list) else value, skip_nulls=False),
)


print(len(df_train_applprev_1))
print(len(result_df))

joined_df = app_prev1.join( ## move up
    result_df,
    on=["case_id", "num_group1"],
    how="inner"  # You can change the join type to "left", "right", "outer", or "cross" as needed
)
joined_df.write_parquet('app_prev.parquet')

print(joined_df)