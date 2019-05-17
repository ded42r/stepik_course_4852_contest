import pandas as pd
from sklearn.metrics import confusion_matrix


def get_tp_fp_fn_tn(y_true, y_pred):
    """ Возвращает массив значений tp, fp, fn, tn

    Parameters
    ----------
    y_true : array
        верные(реальные) метки 
    y_pred : array
        предсказанные метки 
    Returns
    -------
    tp, fp, fn, tn : int
    """
    matrix_y = confusion_matrix(y_true, y_pred)
    tp = matrix_y[1][1]
    fn = matrix_y[1][0]
    fp = matrix_y[0][1]
    tn = matrix_y[0][0]
    return tp, fp, fn, tn


def get_feature_importances_df(feature_importances, features):
    """ Получить датафрейм из важных признаков модели

    Parameters
    ----------
    feature_importances: list if float
        список коэффциентов важностей признаков
    features: list of string
        Список названий признаков

    Returns
    -------
        pd.DataFrame
    """
    fimp = pd.DataFrame([feature_importances], columns=features).T
    fimp.columns = ['weight']
    return fimp.sort_values('weight', ascending=False)
