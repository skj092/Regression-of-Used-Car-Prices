import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import Lasso
import logging
logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level (DEBUG, INFO, WARNING, etc.)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("output.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Load dataset
path = Path('data')
df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')

# Original columns of data: Treat them all as categorical
cols = list(df.columns[1:-1])

# Bi-Grams
logging.info("Generating bi-grams...")
tik = time.time()
new_columns = {}  # for train
new_columns2 = {}  # for test
cols2 = []
for i, c1 in enumerate(cols[:-1]):
    for j, c2 in enumerate(cols[i+1:]):
        name = f"{c1}-{c2}"
        new_columns[name] = df[c1].astype('str') + '-' + df[c2].astype('str')
        new_columns2[name] = test_df[c1].astype(
            'str') + '-' + test_df[c2].astype('str')
        cols2.append(name)

df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(new_columns2)], axis=1)
tok = time.time()
logging.info(f"Time taken for bi-grams: {tok-tik:.2f}s")

# Tri-Grams
logging.info("Generating tri-grams...")
tik = time.time()
new_columns = {}  # for train
new_columns2 = {}  # for test
cols3 = []
for i, c1 in enumerate(cols[:-2]):
    for j, c2 in enumerate(cols[i+1:-1]):
        for k, c3 in enumerate(cols[i+j+2:]):
            name = f"{c1}-{c2}-{c3}"
            new_columns[name] = df[c1].astype(
                'str') + '-' + df[c2].astype('str') + '-' + df[c3].astype('str')
            new_columns2[name] = test_df[c1].astype(
                'str') + '-' + test_df[c2].astype('str') + '-' + test_df[c3].astype('str')
            cols3.append(name)

df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(new_columns2)], axis=1)
tok = time.time()
logging.info(f"Time taken for tri-grams: {tok-tik:.2f}s")

# Target Encoding
target = "price"
target_encode = [f"{c}-TE" for c in cols+cols2+cols3]
more_train = pd.DataFrame(data=np.zeros(
    (len(df), len(target_encode))), columns=target_encode)
more_test = pd.DataFrame(data=np.zeros(
    (len(test_df), len(target_encode))), columns=target_encode)
df = pd.concat([df, more_train], axis=1)
test_df = pd.concat([test_df, more_test], axis=1)

# Fit target encoding on 5 fold to avoid data leakage
kf = KFold(n_splits=5, random_state=42, shuffle=True)
oof = np.zeros(len(df))
pred = np.zeros(len(test_df))

for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    logging.info(f"## Fold {fold}")

    X_train, X_valid = df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
    y_train, y_valid = df[target].iloc[train_idx], df[target].iloc[valid_idx]
    X_test = test_df.copy()

    # Target encoding
    for j, te_col in enumerate(target_encode):
        orig_col = te_col.replace('-TE', "")
        encoder = TargetEncoder(smooth=5)

        # Important: Pass DataFrame instead of Series
        tik = time.time()
        X_train[te_col] = encoder.fit_transform(X_train[[orig_col]], y_train)
        X_valid[te_col] = encoder.transform(
            X_valid[[orig_col]])  # Pass DataFrame with [[]]
        X_test[te_col] = encoder.transform(
            X_test[[orig_col]])    # Pass DataFrame with [[]]
        tok = time.time()
        logging.info(
            f"Time taken for target encoding {j+1}/{len(target_encode)}: {tok-tik:.2f}s")

        # Standardize Encoding
        mean = X_train[te_col].mean()
        std = X_train[te_col].std()

        X_train[te_col] = (X_train[te_col] - mean) / std
        X_valid[te_col] = (X_valid[te_col] - mean) / std
        X_test[te_col] = (X_test[te_col] - mean) / std

    # Select only target encoded features
    X_train = X_train[target_encode]
    X_valid = X_valid[target_encode]
    X_test = X_test[target_encode]

    # Train model
    tik = time.time()
    model = Lasso(alpha=1e2)
    model.fit(X_train, y_train)
    tok = time.time()
    logging.info(f"Time taken for model training: {tok-tik:.2f}s")

    # Make predictions
    oof[valid_idx] = model.predict(X_valid)
    if fold == 0:
        pred = model.predict(X_test)
    else:
        pred += model.predict(X_test)

pred /= 5

# Calculate overall CV score
rmse = np.sqrt(np.mean((oof - df[target].values)**2))
logging.info('Overall CV RMSE:', rmse)

# Save predictions
pd.DataFrame({'id': test_df['id'], 'price': pred}).to_csv(
    path/'submission.csv', index=False)
# Close the file after the script is complete
