import numpy as np
import pandas as pd

from libs import data_helpers as dh, config as conf


def gen_user_step_scores(events, submissions):
    interact_df = dh.create_interaction(events, submissions)
    interact_df.action = interact_df.action.astype('str')

    # расчет весов шагов
    steps_weight_fname = f"{conf.PROCESSED_DATA_DIR}/hb_steps_weight.csv.zip"
    sw_col_name = 'step_weight'
    try:
        hard_steps_weight = pd.read_csv(steps_weight_fname, index_col='step_id')[sw_col_name]
    except FileNotFoundError:
        hard_steps = interact_df.pivot_table(
            index='step_id',
            columns='action',
            values='user_id',
            aggfunc=lambda x: len(np.unique(x)))
        hard_steps_weight = hard_steps.passed / hard_steps.discovered
        hard_steps_weight = hard_steps_weight.rename(sw_col_name)
        hard_steps_weight.to_csv(steps_weight_fname, header=True, compression='zip')

    # расчет баллов за прохождение задания
    data_transform = dh.truncate_data_by_nday(interact_df, conf.DATA_PERIOD_DAYS)
    step_stat = data_transform.pivot_table(
        index=['user_id', 'step_id'],
        columns='action',
        values='timestamp',
        aggfunc='count')
    step_user_scores = step_stat.passed.unstack() / hard_steps_weight
    step_user_scores.columns = ['score_{}'.format(col) for col in step_user_scores.columns]
    step_user_scores = step_user_scores.fillna(0)

    return step_user_scores
