# Rain in Australia

This project predicts whether an Australian weather station will record more than 1 mm of rain in the 24 hours following 09:00 local time. It compares three simple baselines with logistic regression, random forest, LightGBM, CatBoost, and a blend of the two boosting models.

## Setup

The commands below use Windows PowerShell and 64-bit Python 3.13.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name rain-project --display-name "Rain project"
.\.venv\Scripts\python.exe -m jupyterlab
```

The notebooks expect the dataset at `data/weatherAUS.csv`. The CSV is included in the repository. The original dataset is available on [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).

## Run the notebooks

In JupyterLab, select the **Rain project** kernel. 

| Order | Notebook | What it covers |
|---|---|---|
| 1 | `EDA.ipynb` | Data quality, missing values, target distribution, and chronological splits |
| 2 | `models/baseline.ipynb` | Three reference forecasts: always dry, persistence based on `RainToday`, and station–month climatology |
| 3 | `models/LogisticRegression.ipynb` | Logistic regression |
| 4 | `models/RandomForest_rain.ipynb` | Random forest tuning |
| 5 | `models/boosting.ipynb` | LightGBM, CatBoost, their probability blend |

Data cleaning and chronological splits are defined in `src/data_prep.py`. Shared model features are built in `src/features.py`.

