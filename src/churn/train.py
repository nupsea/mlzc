import pandas as pd

import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


C = 1.0
n_splits = 5
output_file = f"model_C={C}.bin"


# Data read and prep
print(f"Reading data..")
df = pd.read_csv("chap3/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Standardize cols and col string data
print("Preparing data")
df.columns = df.columns.str.lower().str.replace(" ", "_")
categorical = list(df.dtypes[df.dtypes == "object"].index)
for c in categorical:
    df[c] = df[c].str.lower().str.replace(" ", "_")

# Fix total-charges
df.totalcharges = pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges = df.totalcharges.fillna(0)

# Convert churn to numerical
df.churn = (df.churn == "yes").astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=20 / 80, random_state=1)
print(
    f"Split DF len (Full Train, Train, Val, Test): {len(df_full_train), len(df_train), len(df_val), len(df_test)}"
)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.pop("churn")
y_train = df_train.pop("churn")
y_val = df_val.pop("churn")
y_test = df_test.pop("churn")
print(
    f"Y DF len (Full Train, Train, Val, Test): {len(y_full_train), len(y_train), len(y_val), len(y_test)}"
)

numerical = ["tenure", "monthlycharges", "totalcharges"]

categorical = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
]


def train(df_t, y, C=1.0):
    dicts = df_t[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_t = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_t, y)

    return dv, model


def predict(df_v, dv, model):
    dicts = df_v[categorical + numerical].to_dict(orient="records")

    X_val = dv.transform(dicts)
    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred


print("Training with KFlod")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 1
for train_idx, val_idx in tqdm(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = y_full_train.iloc[train_idx]
    y_val = y_full_train.iloc[val_idx]

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    score = roc_auc_score(y_val, y_pred)
    print(f"  Fold {fold}: AUC score: {score} ")
    scores.append(score)
    fold += 1


dv, model = train(df_full_train, y_full_train.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print(f"Final model AUC score: {auc}")


with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model output file: {output_file}")
