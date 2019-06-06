import featuretools as ft
import pandas as pd
from featuretools import variable_types as vtypes

import libs.data_helpers as dh
import libs.utils.df_utils as dfu
from libs.config import DATA_PERIOD_DAYS

# описание атрибутов сущности события пользователя по курсу( это обучающие данные)
events_vtypes = {'step_id': vtypes.Id,
                 'user_id': vtypes.Id,
                 'action': vtypes.Categorical,
                 'date': vtypes.Datetime
                 }

# описание атрибутов сущностей о курсе (дополнительный датасет с информацией о курсе)
course_vtypes = {
    'step_id': vtypes.Id,
    'step_correct_ratio': vtypes.Numeric,
    'step_discussions_count': vtypes.Numeric,
    'lesson_id': vtypes.Id,
    'step_max_submissions_count': vtypes.Numeric,
    'step_passed_by': vtypes.Numeric,
    'step_position': vtypes.Ordinal,
    'step_solutions_unlocked_attempts': vtypes.Numeric,
    'step_variation': vtypes.Numeric,
    'step_variations_count': vtypes.Numeric,
    'step_viewed_by': vtypes.Numeric,
    'step_block.options.code_templates_header_lines_count.r': vtypes.Numeric,
    'step_block.options.execution_memory_limit': vtypes.Numeric,
    'step_block.options.execution_time_limit': vtypes.Numeric,
    'step_block.options.limits.r.memory': vtypes.Numeric,
    'step_block.options.limits.r.time': vtypes.Numeric,
    'step_block.video.duration': vtypes.Numeric,
    'assignment_id': vtypes.Id,
    'unit_id': vtypes.Id,
    'unit_position': vtypes.Ordinal,
    'section_id': vtypes.Id,
    'section_position': vtypes.Ordinal,
    'lesson_abuse_count': vtypes.Numeric,
    'lesson_discussions_count': vtypes.Numeric,
    'lesson_epic_count': vtypes.Numeric,
    'lesson_passed_by': vtypes.Numeric,
    'lesson_time_to_complete': vtypes.Numeric,
    'lesson_viewed_by': vtypes.Numeric,
    'lesson_vote_delta': vtypes.Numeric,
    'step_has_submissions_restrictions': vtypes.Boolean,
    'step_is_solutions_unlocked': vtypes.Boolean,
    'step_worth': vtypes.Boolean,
    'step_actions.submit_#': vtypes.Boolean,
    'step_block.name': vtypes.Categorical,
    'step_block.options.is_multiple_choice': vtypes.Categorical,
    'step_block.options.is_run_user_code_allowed': vtypes.Categorical,
    'section_title': vtypes.Categorical,
    'lesson_title': vtypes.Categorical
}


def prepare_ft(events, submissions, hb_course_df, n_users_sample=None):
    """ подготовить данные и создать представление сущностей и связей """
    interactions = dh.create_interaction(events, submissions)
    interactions = dh.preprocess_timestamp_cols(interactions)
    dfu.safe_drop_cols_df(interactions, ['day', 'timestamp'])

    # сделаем случайную подвыборку пользователей для которых будем
    if n_users_sample is None:  # все пользователи
        user_rand = None
    else:
        user_rand = interactions.user_id.sample(n_users_sample).unique()
        interactions = (interactions.query('user_id in @user_rand')
                        .reindex())

    # формируем сущности для featurestools
    es = create_es(interactions, hb_course_df)
    cut_off_time = create_cut_off_time(interactions, es, user_rand)
    return es, cut_off_time


def create_es(interactions_train, course_df):
    """ создание представления сущностей для featuretools """
    es = ft.EntitySet('user_events')
    es = es.entity_from_dataframe(entity_id="events",
                                  dataframe=interactions_train.copy(),
                                  make_index=True,
                                  index='id',
                                  time_index='date',
                                  variable_types=events_vtypes)
    es = es.entity_from_dataframe(entity_id="steps",
                                  dataframe=course_df.copy(),
                                  index='step_id',
                                  variable_types=course_vtypes)

    es.normalize_entity('events', 'users', 'user_id', make_time_index=False);
    es = es.add_relationship(ft.Relationship(es['steps']['step_id'], es['events']['step_id']))

    lesson_additional_variables = ['lesson_abuse_count', 'lesson_discussions_count', 'lesson_epic_count',
                                   'lesson_passed_by', 'lesson_time_to_complete', 'lesson_title',
                                   'lesson_viewed_by', 'lesson_vote_delta',
                                   'section_id', 'section_position', 'section_title']
    es.normalize_entity('steps', 'lessons', 'lesson_id',
                        additional_variables=lesson_additional_variables,
                        make_time_index=False);

    sections_additional_variables = ['section_position', 'section_title']
    es.normalize_entity('lessons', 'sections', 'section_id',
                        additional_variables=sections_additional_variables,
                        make_time_index=False);

    es["events"]["action"].interesting_values = interactions_train.action.unique().categories
    es["steps"]["step_block.name"].interesting_values = course_df['step_block.name'].unique()

    return es


def create_cut_off_time(interactions_train, es, user_rand=None, day_offset=DATA_PERIOD_DAYS):
    """ создать порог отсечения для каждого пользователя. 
    featurestools будет отбрасывать все события после этой даты(для каждого пользователя своя) """
    cut_off_time = interactions_train.groupby('user_id')['date'].min() + pd.DateOffset(days=day_offset)
    cut_off_time = cut_off_time.rename('last_date').to_frame()
    cut_off_time = pd.merge(es['users'].df, cut_off_time, how='outer', left_on='user_id', right_index=True)

    if user_rand is not None:
        cut_off_time = cut_off_time.query('user_id in @user_rand')

    return cut_off_time


def create_features(es, cut_off_time, chunk_size=.05, n_jobs=3):
    """ создание признаков  """
    agg_primitives = [
        'num_unique', 'count', 'percent_true',
        'avg_time_between', 'time_since_first', 'time_since_last',
        'trend', 'last', 'mean', 'min', 'max', 'std', 'mode', 'skew',
        'median', 'num_unique', 'sum',
    ]
    trans_primitives = [
        'is_weekend',
        'days_since', 'time_since', 'time_since_previous', 'day',
        'weekday', 'month'
    ]
    where_primitives = [
        "count", 'percent_true', 'mean',
        "count", "avg_time_between", 'time_since_first',
        'time_since_last', 'percent_true', 'trend',
    ]
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_entity='users',
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        where_primitives=where_primitives,
        max_depth=3,
        cutoff_time=cut_off_time,
        features_only=False,
        n_jobs=n_jobs, chunk_size=chunk_size,
        approximate="6 hour",
        verbose=True)
    return feature_matrix, feature_defs
