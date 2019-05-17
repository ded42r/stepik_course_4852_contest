import collections

import numpy as np


def safe_drop_cols_df(df, drop_cols):
    """ Удаление столбцов из датафрейма с проверкой на их существование.
    Изменяет переданный датафрейм
    
    Parameters
    ----------
    df: pandas.DataFrame
        датафрейм
    drop_cols: list of string
        Список столбцов которые нужно удалить
    """

    if isinstance(drop_cols, str) or (not isinstance(drop_cols, collections.Iterable)):
        drop_cols = [drop_cols]
    drop_col_names = np.intersect1d(df.columns, drop_cols)
    df.drop(drop_col_names, axis=1, inplace=True)


def reorder_cols_df(df, first_cols):
    """ изменить порядок столбцов, так чтобы сперва были переданные
    
    Parameters
    ----------
    df: pandas.DataFrame
    first_cols: list of string
        Список столбцов которые нужно поставить первыми    
    """
    other_cols = np.setdiff1d(df.columns, first_cols)
    new_cols = np.concatenate((first_cols, other_cols))
    return df[new_cols]


def split_df_chunk(df, chunkSize=100):
    """ разделить датафрейм на чанки 
    
    Parameters
    ----------
    df: pandas.DataFrame
    chunkSize: int
        колво строк в чанке    
    """
    numberChunks = len(df) // chunkSize + 1
    return np.array_split(df, numberChunks, axis=0)
