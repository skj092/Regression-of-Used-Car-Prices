from cuml import Lasso
from cuml.preprocessing.TargetEncoder import TargetEncoder
from sklearn.model_selection import KFold
from pathlib import Path
import numpy as np
import cudf
import time
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
VER = 1

# Load dataset
path = Path("/kaggle/input/playground-series-s4e9")
df = cudf.read_csv(path / "train.csv")
test_df = cudf.read_csv(path / "test.csv")

# Original columns of data: Treat them all as categorical
cols = list(df.columns[1:-1])

# Bi-Grams
tik = time.time()
new_columns = {}  # for train
new_columns2 = {}  # for test
cols2 = []
for i, c1 in enumerate(cols[:-1]):
    for j, c2 in enumerate(cols[i + 1:]):
        name = f"{c1}-{c2}"
        new_columns[name] = df[c1].astype("str") + "-" + df[c2].astype("str")
        new_columns2[name] = test_df[c1].astype(
            "str") + "-" + test_df[c2].astype("str")
        cols2.append(name)

df = cudf.concat([df, cudf.DataFrame(new_columns)], axis=1)
test_df = cudf.concat([test_df, cudf.DataFrame(new_columns2)], axis=1)
tok = time.time()
print(f"Time taken for bi-grams: {tok-tik:.2f}s")

# Tri-Grams
print("Generating tri-grams...")
tik = time.time()
new_columns = {}  # for train
new_columns2 = {}  # for test
cols3 = []
for i, c1 in enumerate(cols[:-2]):
    for j, c2 in enumerate(cols[i + 1: -1]):
        for k, c3 in enumerate(cols[i + j + 2:]):
            name = f"{c1}-{c2}-{c3}"
            new_columns[name] = (
                df[c1].astype("str")
                + "-"
                + df[c2].astype("str")
                + "-"
                + df[c3].astype("str")
            )
            new_columns2[name] = (
                test_df[c1].astype("str")
                + "-"
                + test_df[c2].astype("str")
                + "-"
                + test_df[c3].astype("str")
            )
            cols3.append(name)

df = cudf.concat([df, cudf.DataFrame(new_columns)], axis=1)
test_df = cudf.concat([test_df, cudf.DataFrame(new_columns2)], axis=1)
tok = time.time()
print(f"Time taken for tri-grams: {tok-tik:.2f}s")

# Target Encoding
target = "price"
target_encode = [f"{c}-TE" for c in cols + cols2 + cols3]
more_train = cudf.DataFrame(
    data=np.zeros((len(df), len(target_encode))), columns=target_encode
)
more_test = cudf.DataFrame(
    data=np.zeros((len(test_df), len(target_encode))), columns=target_encode
)
df = cudf.concat([df, more_train], axis=1)
test_df = cudf.concat([test_df, more_test], axis=1)

# Fit target encoding on 5 fold to avoid data leakage
kf = KFold(n_splits=5, random_state=42, shuffle=True)
oof = np.zeros(len(df))
pred = np.zeros(len(test_df))

for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    print(f"## Fold {fold}")

    X_train, X_valid = df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
    y_train, y_valid = df[target].iloc[train_idx], df[target].iloc[valid_idx]
    X_test = test_df.copy()

    # Target encoding
    for j, te_col in enumerate(tqdm(target_encode)):
        orig_col = te_col.replace("-TE", "")
        encoder = TargetEncoder(smooth=5)

        # Important: Pass DataFrame instead of Series
        X_train[te_col] = encoder.fit_transform(X_train[[orig_col]], y_train)
        X_valid[te_col] = encoder.transform(
            X_valid[[orig_col]]
        )  # Pass DataFrame with [[]]
        X_test[te_col] = encoder.transform(
            X_test[[orig_col]]
        )  # Pass DataFrame with [[]]

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
    print(f"Time taken for model training: {tok-tik:.2f}s")

    # Make predictions
    oof[valid_idx] = model.predict(X_valid).to_numpy()
    if fold == 0:
        pred = model.predict(X_test).to_numpy()
    else:
        pred += model.predict(X_test).to_numpy()

pred /= 5

# Calculate overall CV score
rsme = np.sqrt(np.mean((oof - df.price.to_numpy()) ** 2))
print("Overall CV RMSE:", rsme)

# Save predictions
sub = cudf.read_csv(path / "sample_submission.csv")
sub["id"] = test_df["id"]
sub["price"] = pred
sub.to_csv("submission.csv", index=False)
