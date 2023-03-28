import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from dateutil.relativedelta import relativedelta


def make_ds_add_new(
    Data,
    start,
    end,
    enc,
    window=10,
    to_horizon=30,
    delta_in_x=15,
    station="Барабинск",
    added_feature=["Температура воздуха по сухому термометру"],
):
    """
    Makes a dataset consisting of X and y for a special meteostation. With given values, if we consider 01.01,
    we take 01.01-15.01 in X (delta_in_x=15), look for the exceedance of 20 m/sec in 20.01-30.01 (window=10 and to_horizon=30).

    start - start date of the period, for example pd.to_datetime('1990-01-01');
    end - end date of the period, for example pd.to_datetime('1993-01-31'), dataset will be gathered to 1993-01-01 as to_horizon=30;
    enc - fitted OneHotEncoder, for example on day_of_year;
    window - period in the near future in which we search for the wind speed exceedance;
    to_horizon - how many days we look ahead in search of wind speed exceeds;
    delta_in_x - how many days are included into X, always less than to_horizon;
    """
    X, y = [], []
    added_feature = ["Максимальная скорость"] + added_feature
    station = Data.loc[Data["Название метеостанции"] == station][added_feature]
    station_train = station.loc[
        (station.index >= start)
        & (station.index <= end + datetime.timedelta(to_horizon))
    ]
    station_train = station_train.groupby(station_train.index).max()
    y_ind = np.where(
        station_train.rolling(window=str(window) + "D")["Максимальная скорость"].max()
        >= 20
    )[0] - (
        to_horizon
    )  # - to_horizon
    y = np.array([0] * (station_train.shape[0] - (to_horizon)))
    y_ind = [x for x in y_ind if x >= 0]
    y[y_ind] = 1
    # lis = [[station_train.iloc[i:i+delta_in_x], station_train.iloc[i+delta_in_x].name.dayofyear] for i in range(station_train.shape[0] - (to_horizon - delta_in_x))]
    lis = [
        [
            np.concatenate(
                (
                    np.array(station_train.iloc[i : i + delta_in_x]).reshape(
                        1, delta_in_x
                    ),
                    enc.transform(
                        np.atleast_2d(station_train.iloc[i + delta_in_x].name.dayofyear)
                    ).toarray(),
                ),
                axis=1,
            )
        ]
        for i in range(station_train.shape[0] - (to_horizon))
    ]
    # station_train.iloc[i:i+delta_in_x] - it is 15 (delta_in_x) days included in X, need to reshape it for concatenation
    # enc.transform(np.atleast_2d(station_train.iloc[i+delta_in_x].name.dayofyear)).toarray() - that is a OneHotEncoded day of year
    X = np.array(lis).reshape(np.array(lis).shape[0], -1)

    return X, y


def make_ds_add_feat(
    Data,
    start,
    end,
    enc,
    window=10,
    to_horizon=30,
    delta_in_x=15,
    station="Барабинск",
    added_feature=["Температура воздуха по сухому термометру"],
):
    """
    Makes a dataset consisting of X and y for a special meteostation. With given values, if we consider 01.01,
    we take 01.01-15.01 in X (delta_in_x=15), look for the exceedance of 20 m/sec in 20.01-30.01 (window=10 and to_horizon=30).

    start - start date of the period, for example pd.to_datetime('1990-01-01');
    end - end date of the period, for example pd.to_datetime('1993-01-31'), dataset will be gathered to 1993-01-01 as to_horizon=30;
    enc - fitted OneHotEncoder, for example on day_of_year;
    window - period in the near future in which we search for the wind speed exceedance;
    to_horizon - how many days we look ahead in search of wind speed exceeds;
    delta_in_x - how many days are included into X, always less than to_horizon;
    """
    X, y = [], []
    added_feature = ["Максимальная скорость"] + added_feature
    station = Data.loc[Data["Название метеостанции"] == station][added_feature]
    station_train = station.loc[
        (station.index >= start)
        & (station.index <= end + datetime.timedelta(to_horizon))
    ]
    station_train = station_train.groupby(station_train.index).max()
    y_ind = np.where(
        station_train.rolling(window=str(window) + "D")["Максимальная скорость"].max()
        >= 20
    )[0] - (
        to_horizon - delta_in_x
    )  # - to_horizon
    y = np.array([0] * (station_train.shape[0] - (to_horizon - delta_in_x)))
    y_ind = [x for x in y_ind if x >= 0]
    y[y_ind] = 1
    # lis = [[station_train.iloc[i:i+delta_in_x], station_train.iloc[i+delta_in_x].name.dayofyear] for i in range(station_train.shape[0] - (to_horizon - delta_in_x))]
    lis = [
        [
            np.concatenate(
                (
                    np.array(station_train.iloc[i : i + delta_in_x]).reshape(
                        1, delta_in_x
                    ),
                    enc.transform(
                        np.atleast_2d(station_train.iloc[i + delta_in_x].name.dayofyear)
                    ).toarray(),
                ),
                axis=1,
            )
        ]
        for i in range(station_train.shape[0] - (to_horizon - delta_in_x))
    ]
    # station_train.iloc[i:i+delta_in_x] - it is 15 (delta_in_x) days included in X, need to reshape it for concatenation
    # enc.transform(np.atleast_2d(station_train.iloc[i+delta_in_x].name.dayofyear)).toarray() - that is a OneHotEncoded day of year
    X = np.array(lis).reshape(np.array(lis).shape[0], -1)

    return X, y


def make_ds_add_year(
    Data,
    start,
    end,
    enc,
    window=10,
    to_horizon=30,
    delta_in_x=15,
    station="Барабинск",
    added_feature=["Температура воздуха по сухому термометру"],
):
    """
    Makes a dataset consisting of X and y for a special meteostation. With given values, if we consider 01.01,
    we take 01.01-15.01 in X (delta_in_x=15), look for the exceedance of 20 m/sec in 20.01-30.01 (window=10 and to_horizon=30).

    start - start date of the period, for example pd.to_datetime('1990-01-01');
    end - end date of the period, for example pd.to_datetime('1993-01-31'), dataset will be gathered to 1993-01-01 as to_horizon=30;
    enc - fitted OneHotEncoder, for example on day_of_year;
    window - period in the near future in which we search for the wind speed exceedance;
    to_horizon - how many days we look ahead in search of wind speed exceeds;
    delta_in_x - how many days are included into X, always less than to_horizon;
    """
    X, y = [], []
    added_feature = ["Максимальная скорость"] + added_feature
    station = Data.loc[Data["Название метеостанции"] == station][added_feature]
    station_train = station.loc[
        (station.index >= start)
        & (
            station.index
            <= end + datetime.timedelta(to_horizon) + relativedelta(years=1)
        )
    ]
    station_train = station_train.groupby(station_train.index).max()
    y_ind = (
        np.where(
            station_train.rolling(window=str(window) + "D")[
                "Максимальная скорость"
            ].max()
            >= 20
        )[0]
        - (to_horizon - delta_in_x)
        - 365
    )  # - to_horizon
    y = np.array([0] * (station_train.shape[0] - (to_horizon - delta_in_x) - 365))
    y_ind = [x for x in y_ind if x >= 0]
    y[y_ind] = 1
    # lis = [[station_train.iloc[i:i+delta_in_x], station_train.iloc[i+delta_in_x].name.dayofyear] for i in range(station_train.shape[0] - (to_horizon - delta_in_x))]
    lis = [
        [
            np.concatenate(
                (
                    np.array(station_train.iloc[i : i + delta_in_x]).reshape(
                        1, delta_in_x
                    ),
                    enc.transform(
                        np.atleast_2d(station_train.iloc[i + delta_in_x].name.dayofyear)
                    ).toarray(),
                ),
                axis=1,
            )
        ]
        for i in range(station_train.shape[0] - (to_horizon - delta_in_x) - 365)
    ]
    # station_train.iloc[i:i+delta_in_x] - it is 15 (delta_in_x) days included in X, need to reshape it for concatenation
    # enc.transform(np.atleast_2d(station_train.iloc[i+delta_in_x].name.dayofyear)).toarray() - that is a OneHotEncoded day of year
    X = np.array(lis).reshape(np.array(lis).shape[0], -1)

    return X, y


def metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # r_a_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # metrics = [r_a_score, acc, precision, recall, f1]
    metrics = [acc, precision, recall, f1]
    return metrics
