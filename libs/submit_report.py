import datetime

import numpy as np
import pandas as pd


def create_report(user_ids, preds):
    res = pd.DataFrame(preds[:, np.newaxis], columns=['is_gone'], index=user_ids)
    return res.reset_index()


def save_report(rep, submit_num=None):
    submit_str = ''
    if submit_num:
        submit_str = '_submit_{}'.format(submit_num)
    fname = './reports/predict_{:%Y-%m-%d}{}.csv'.format(datetime.datetime.now(), submit_str)
    rep.to_csv(fname, index=False)
    return fname
