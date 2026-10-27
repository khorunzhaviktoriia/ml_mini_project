import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
DATA_PATH = "data/weatherAUS.csv"
TARGET = "RainTomorrow"

TEST_START = pd.Timestamp("2016-07-01")
GAP_DAYS = 1
N_FOLDS = 4
FOLD_DAYS = 365

NUMERIC_COLUMNS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
FEATURES = NUMERIC_COLUMNS + ["RainToday"] + CATEGORICAL_FEATURES


def clean(raw, verbose=False):
    df = raw.dropna(subset=[TARGET]).copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    for col in ["Cloud9am", "Cloud3pm"]:
        invalid = df[col].notna() & ~df[col].between(0, 8)
        if verbose:
            print(f"{col}: {invalid.sum()} values replaced with NaN")
        df.loc[invalid, col] = np.nan

    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype("int8")
    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1}).astype("float32")

    return df.sort_values(["Date", "Location"]).reset_index(drop=True)


def load_clean(path=DATA_PATH):
    return clean(pd.read_csv(path))


def split_train_test(df):
    train_end = TEST_START - pd.Timedelta(days=GAP_DAYS)
    train = df[df["Date"] < train_end].copy()
    gap = df[(df["Date"] >= train_end) &(df["Date"] < TEST_START)].copy()
    test = df[df["Date"] >= TEST_START].copy()

    return train, gap, test


def time_cv_splits(dates, n_folds=N_FOLDS, fold_days=FOLD_DAYS, gap_days=GAP_DAYS):
    dates = pd.Series(pd.to_datetime(dates).to_numpy())
    end, folds = dates.max(), []

    for i in range(n_folds):
        v_end = end - pd.Timedelta(days=fold_days * i)
        v_start = v_end - pd.Timedelta(days=fold_days)
        train_idx = np.flatnonzero((dates <= v_start - pd.Timedelta(days=gap_days)).to_numpy())
        val_idx = np.flatnonzero(((dates > v_start) & (dates <= v_end)).to_numpy())
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {i} is empty: reduce n_folds or fold_days")
        folds.append((train_idx, val_idx))

    return folds[::-1]


def get_x_y(part):
    return part[FEATURES], part[TARGET]


def build_preprocessor():
    return ColumnTransformer(
        [
            ("numeric",SimpleImputer(strategy="median",add_indicator=True),NUMERIC_COLUMNS),
            ("rain_today", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), ["RainToday"]),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
