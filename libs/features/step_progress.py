from itertools import chain

import numpy as np
import pandas as pd

from libs import config as conf


def gen_progress_features(X):
    # расчет фич отношений
    prog_ft = pd.DataFrame(index=X.index)

    prog_ft['correct_rat_attempts'] = X.correct / (X.correct + X.wrong)
    prog_ft['correct_rat_attempts'] = prog_ft['correct_rat_attempts'].fillna(-1)
    return prog_ft


def create_ratio_features_action(users_data):
    dfs = []
    for col in np.setdiff1d(conf.ACTION_CATEGORIES, ['discovered']):
        df_iter = (users_data[col] / users_data['discovered'].replace(0, 1)).rename('{}_rat_discovered'.format(col))
        dfs += [df_iter]
    dfs = pd.concat(dfs, axis=1)
    dfs = dfs.fillna(-1)
    return dfs


def create_ratio_features_action_subm_status(users_data):
    dfs = []
    for scol in conf.SUBMISSION_STATUSES:
        for acol in conf.ACTION_CATEGORIES:
            df_iter = (users_data[scol] / users_data[acol].replace(0, 1)).rename('{}_rat_{}'.format(scol, acol))
            dfs += [df_iter]
    dfs = pd.concat(dfs, axis=1)
    dfs = dfs.fillna(-1)
    return dfs


def create_ratio_features_day(users_data):
    """ сгенерировать признаки  """
    dfs = []
    for col in chain(conf.ACTION_CATEGORIES, conf.SUBMISSION_STATUSES):
        df_iter = (users_data[col] / users_data.day).rename('{}_rat_day'.format(col))
        dfs += [df_iter]
    dfs = pd.concat(dfs, axis=1)
    dfs = dfs.fillna(-1)
    return dfs
