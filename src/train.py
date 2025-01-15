from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder


# load dataset
path = Path('data')
df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')

# collect categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=np.number).drop(
    columns=['price']).columns.tolist()


# handle missing cols
def fill_missing(df, cat_cols):
    for col in df.columns:
        if col in cat_cols:
            df[col] = df[col].fillna(df[col].mode().values[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


df = fill_missing(df, cat_cols)

# handle categorical columns
df[cat_cols] = df[cat_cols].astype('category')
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

    if col in test_df:
        test_df[col] = test_df[col].map(
            lambda s: '<unknown>' if s not in le.classes_ else s)


# cross validation
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)

tik = time.monotonic()
model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
error = root_mean_squared_error(y_pred, y_valid)
tok = time.monotonic()
print(f"CV Score : {error:.2f} | Time : {tok-tik:.2f}s")
