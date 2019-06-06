import datetime

import numpy as np
import pandas as pd

import libs.config as conf


def create_report(user_ids, preds):
    res = pd.DataFrame(preds[:, np.newaxis], columns=['is_gone'], index=user_ids)
    return res.reset_index()


def save_report(rep, submit_num=None):
    submit_str = ""
    if submit_num:
        submit_str = f"_submit_{submit_num}"
    fname = f"{conf.REPORTS_DIR}/predict_{datetime.datetime.now():%Y-%m-%d}{submit_str}.csv"
    rep.to_csv(fname, index=False)
    return fname
