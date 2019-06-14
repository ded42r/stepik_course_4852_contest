import pandas as pd

import libs.data_helpers as dh
import libs.features.step_progress as fsp
from libs import data_iter1 as di1
from libs import data_iter_auto as di_auto
from libs.features.step_weight import gen_user_step_scores


def get_x_y(events, submissions):
    X, y = di_auto.get_x_y(events, submissions)

    # полуручные признаки по степам (взаимодействие одних событий с другими
    x_iter1, _ = di1.get_x_y(events, submissions)
    func_gen_features = (fsp.gen_progress_features, fsp.create_ratio_features_action,
                         fsp.create_ratio_features_action_subm_status, fsp.create_ratio_features_day)
    interact_features = [gen_fun(x_iter1) for gen_fun in func_gen_features]
    interact_features = pd.concat(interact_features, axis=1)
    interact_features = interact_features.fillna(0)
    X = pd.concat([X, interact_features], axis=1)

    # признаки сгенеренные featuretools
    user_step_scores = gen_user_step_scores(events, submissions)
    user_step_scores = dh.make_intersect_by_users(events, submissions, user_step_scores)
    #  Отбирал важныепризнаки с помощью boruta
    user_step_scores = user_step_scores[
        ['score_31971', 'score_31972', 'score_31976', 'score_31977',
         'score_31978', 'score_32031', 'score_32173', 'score_32174',
         'score_32175', 'score_32177', 'score_32219', 'score_32812',
         'score_32815', 'score_32929', 'score_32950']]
    X = X.merge(user_step_scores, how='left', left_index=True, right_index=True, validate='1:1')

    assert X.shape[0] == y.shape[0]
    return X, y
