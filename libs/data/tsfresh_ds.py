import numpy as np
import pandas as pd
import tsfresh


def gen_fc_params():
    """ параметры для генерации признаков из временных рядов"""
    fcp_params = tsfresh.feature_extraction.EfficientFCParameters()
    final_params = {}
    for k in ['autocorrelation', 'c3', 'kurtosis', 'length', 'maximum', 'mean',
              'median', 'minimum', 'skewness', 'variance']:
        final_params[k] = fcp_params[k]
    return final_params


def prep_ts_interact(data):
    """ подготовка данных, представляем действия пользователя как временной ряд """
    interact_sub = data[['user_id', 'date', 'action']]
    interact_sub['weight'] = interact_sub.action.cat.codes
    interact_sub.action = interact_sub.action.astype('str')

    ts_df = pd.pivot_table(interact_sub, index=['user_id', 'date'], columns='action',
                           values='weight', aggfunc=np.max)
    ts_df = ts_df.reset_index()
    ts_df = ts_df.sort_values('user_id')
    ts_df = ts_df.fillna(0)
    return ts_df


def gen_ts_features(ts_data):
    """ сгенерировать датасет с рпизнаками """
    TSF_PARAMS = {
        'chunksize': 5,
        'n_jobs': 2,
        'fc_params': gen_fc_params()
    }
    tsf_df_test = tsfresh.extract_features(
        ts_data,
        column_id='user_id',
        column_sort='date',
        default_fc_parameters=TSF_PARAMS['fc_params'],
        chunksize=TSF_PARAMS['chunksize'],
        disable_progressbar=False,
        n_jobs=TSF_PARAMS['n_jobs'])
    tsf_df_test.index = tsf_df_test.index.rename('user_id')
    return tsf_df_test
