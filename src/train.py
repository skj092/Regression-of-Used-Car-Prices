from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error


# load dataset
path = Path('data')
df = pd.read_csv(path/'train.csv')

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

# cross validation
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
error = root_mean_squared_error(y_pred, y_valid)
print(f"score is {error}")
