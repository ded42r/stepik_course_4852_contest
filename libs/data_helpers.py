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


def create_interaction(events, submissions):
    """ объединить все данные по взаимодействию

    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователя
    submissions : pd.DataFrame
        данные самбитов практики
    """
    interact_train = pd.concat([events, submissions.rename(columns={'submission_status': 'action'})])
    interact_train.action = pd.Categorical(interact_train.action,
                                           ['discovered', 'viewed', 'started_attempt',
                                            'wrong', 'passed', 'correct'], ordered=True)
    interact_train = interact_train.sort_values(['user_id', 'timestamp', 'action'])
    return interact_train


def create_user_data(events, submissions):
    """ создать таблицу с данными по каждому пользователю

    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователя
    submissions : pd.DataFrame
        данные самбитов практики
    """
    users_data = events.groupby('user_id', as_index=False) \
        .agg({'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})

    # попытки сдачи практики пользователя
    users_scores = submissions.pivot_table(index='user_id',
                                           columns='submission_status',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
    users_data = users_data.merge(users_scores, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    # колво разных событий пользователя по урокам
    users_events_data = events.pivot_table(index='user_id',
                                           columns='action',
                                           values='step_id',
                                           aggfunc='count',
                                           fill_value=0).reset_index()
    users_data = users_data.merge(users_events_data, how='outer')

    # колво дней на курсе
    users_days = events.groupby('user_id').day.nunique().to_frame().reset_index()
    users_data = users_data.merge(users_days, how='outer')

    return users_data


def get_y(events, submissions, course_threshold=40, target_action='correct'):
    """ создать метку  (целевая переменная для прогноза is_gone

    Parameters
    ----------
    events : pd.DataFrame
        данные с действиями пользователя
    submissions : pd.DataFrame
        данные самбитов практики
    course_threshold : int
        порог в колве заданий, когда курс считается пройденным
    target_action: string
        название действия по степу, по колву которых мы рассчитываем целевую переменную 
    """
    interactions = create_interaction(events, submissions)
    users_data = interactions[['user_id']].drop_duplicates()

    assert target_action in interactions.action.unique()

    # вместо count по хорошему нужно брать уникальные степы. Потому что correct может встречаться более 1 раза
    passed_steps = (interactions.query("action == @target_action")
                    .groupby('user_id', as_index=False)['step_id'].agg(lambda a: len(np.unique(a)))
                    .rename(columns={'step_id': target_action}))
    users_data = users_data.merge(passed_steps, how='outer')

    # пройден ли курс
    users_data['is_gone'] = users_data[target_action] > course_threshold
    assert users_data.user_id.nunique() == events.user_id.nunique()
    users_data = (users_data.drop(target_action, axis=1)
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


def make_intersect_by_users(events, submissions, data):
    """ вернуть данные из data только  по тем пользователям, которые есть в events, submissions """
    user_ids = np.unique(np.concatenate((events.user_id.unique(),
                                         submissions.user_id.unique())))

    # проверяем что для всех пользователей которых передали есть информация
    diff_users = np.setdiff1d(user_ids, data.index.values)
    assert len(diff_users) == 0

    return data.loc[user_ids]
