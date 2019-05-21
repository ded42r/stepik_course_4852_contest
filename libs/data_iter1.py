import libs.data_helpers as dh
from libs.utils.df_utils import safe_drop_cols_df

DATA_PERIOD_DAYS = 2  # колво дней по которым доступны данные для прогноза


def get_x_y(events, submissions):
    """" создадим признаки и метку
     
    Parameters
    ----------
    events: pandas.DataFrame
        действия студентов со степами
    submissions: pandas.DataFrame
        действия студентов по практике     
     """
    events_train = dh.preprocess_timestamp_cols(events)
    events_train = dh.truncate_data_by_nday(events_train, DATA_PERIOD_DAYS)

    submissions_train = dh.preprocess_timestamp_cols(submissions)
    submissions_train = dh.truncate_data_by_nday(submissions_train, DATA_PERIOD_DAYS)

    X = dh.create_user_data(events_train, submissions_train)
    X = X.set_index('user_id')
    safe_drop_cols_df(X, ['last_timestamp'])

    y = dh.get_y(events, submissions)

    # после создания признаков и метки порядок следования user_id может не совпадать
    X = X.sort_index()
    y = y.sort_index()
    assert X.shape[0] == y.shape[0]
    return X, y
