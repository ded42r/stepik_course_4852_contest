import numpy as np
import pandas as pd


def preprocess_timestamp_cols(data):
    """ 
    Parameters
    ----------
    data : pd.DataFrame
        данные с действиями пользователя
    """
    data['date'] = pd.to_datetime(data.timestamp, unit='s')
    data['day'] = data.date.dt.date
    return data


def create_user_data(events_data, submissions_data):
    """ создать таблицу с данными по каждому пользователю

    Parameters
    ----------
    events_data : pd.DataFrame
        данные с действиями пользователя
    submissions_data : pd.DataFrame
        данные самбитов практики
    """
    users_data = events_data.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})

    # попытки сдачи практики пользователя
    users_scores = submissions_data.pivot_table(index='user_id',
                                                columns='submission_status',
                                                values='step_id',
                                                aggfunc='count',
                                                fill_value=0).reset_index()
    users_data = users_data.merge(users_scores, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    # колво разных событий пользователя по урокам
    users_events_data = events_data.pivot_table(index='user_id',
                                                columns='action',
                                                values='step_id',
                                                aggfunc='count',
                                                fill_value=0).reset_index()
    users_data = users_data.merge(users_events_data, how='outer')

    # колво дней на курсе
    users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()
    users_data = users_data.merge(users_days, how='outer')

    return users_data


def get_y(events_data, course_threshold=40):
    """ создать метку  (целевая переменная для прогноза is_gone

    Parameters
    ----------
    events_data : pd.DataFrame
        данные с действиями пользователя
    course_threshold : int
        порог в колве заданий, когда курс считается пройденным
    """
    users_data = events_data[['user_id']].drop_duplicates()

    passed_steps = (events_data.query("action == 'passed'")
                    .groupby('user_id', as_index=False)['step_id'].count()
                    .rename(columns={'step_id': 'passed'}))
    users_data = users_data.merge(passed_steps, how='outer')

    # пройден ли курс
    users_data['is_gone'] = users_data.passed > course_threshold
    assert users_data.user_id.nunique() == events_data.user_id.nunique()
    users_data = (users_data.drop('passed', axis=1)
                  .set_index('user_id'))
    return users_data['is_gone']


def truncate_data_by_nday(data, n_day):
    """ Взять события из n_day первых дней по каждому пользователю 
    
        Parameters
        ----------
        data: pandas.DataFrame
            действия студентов со степами или практикой
        n_day : int
            размер тестовой выборки
    """
    users_min_time = data.groupby('user_id', as_index=False).agg({'timestamp': 'min'}).rename(
        {'timestamp': 'min_timestamp'}, axis=1)
    users_min_time['min_timestamp'] += 60 * 60 * 24 * n_day

    events_data_d = pd.merge(data, users_min_time, how='inner', on='user_id')
    cond = events_data_d['timestamp'] <= events_data_d['min_timestamp']
    events_data_d = events_data_d[cond]

    assert events_data_d.user_id.nunique() == data.user_id.nunique()
    return events_data_d.drop(['min_timestamp'], axis=1)


def split_events_submissions(events, submissions, test_size=0.3):
    """ разделение выборки на трейн и тест по пользователям
     
        Parameters
        ----------
        events: pandas.DataFrame
            действия студентов со степами
        submissions: pandas.DataFrame
            действия студентов по практике     
        test_size : float
            размер тестовой выборки
     """

    # сделаем случайную выборку пользователей курса для теста
    users_ids = np.unique(np.concatenate((events.user_id.unique(), submissions.user_id.unique())))
    np.random.shuffle(users_ids)
    test_sz = int(len(users_ids) * test_size)
    train_sz = len(users_ids) - test_sz
    train_users = users_ids[:train_sz]
    test_users = users_ids[-test_sz:]
    # Проверка что пользователи не пересекаются
    assert len(np.intersect1d(train_users, test_users)) == 0

    # теперь делим данные
    event_train = events[events.user_id.isin(train_users)]
    event_test = events[events.user_id.isin(test_users)]
    submissions_train = submissions[submissions.user_id.isin(train_users)]
    submissions_test = submissions[submissions.user_id.isin(test_users)]

    return event_train, event_test, submissions_train, submissions_test
