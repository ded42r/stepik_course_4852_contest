import pandas as pd

import libs.config as conf


def load_cache_ts_features():
    ts_features_train = pd.read_csv(f"{conf.PROCESSED_DATA_DIR}/ts_features_train.zip")
    ts_features_submit = pd.read_csv(f"{conf.PROCESSED_DATA_DIR}/ts_features_submit.zip")
    all_ts_data = pd.concat([ts_features_train, ts_features_submit])
    all_ts_data = all_ts_data.fillna(all_ts_data.mean())
    all_ts_data = all_ts_data.set_index('user_id')
    return all_ts_data
