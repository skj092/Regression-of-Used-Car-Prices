from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import math
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
test_df = fill_missing(test_df, cat_cols)

# handle categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    if col in test_df.columns:
        # Handle unknown categories in test set
        unique_test_categories = test_df[col].unique()
        unknown_categories = [
            cat for cat in unique_test_categories if cat not in le.classes_]
        if unknown_categories:
            # Add a new class for unknown categories
            new_classes = np.append(le.classes_, ['unknown'])
            le.classes_ = new_classes
            # Replace unknown categories with 'unknown'
            test_df.loc[test_df[col].isin(unknown_categories), col] = 'unknown'
        # Transform test data
        test_df[col] = le.transform(test_df[col])

# cross validation
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)

tik = time.monotonic()
model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
error = math.sqrt(mean_squared_error(y_pred, y_valid))
tok = time.monotonic()
print(f"CV Score : {error:.2f} | Time : {tok-tik:.2f}s")

# Inference
# import code; code.interact(local=locals())
y_test = model.predict(test_df)
submission = pd.DataFrame({'id': test_df['id'], 'price': y_test})
submission.to_csv('submission.csv', index=False)
