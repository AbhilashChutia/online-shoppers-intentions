import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

xgb_params = {
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 1,
    "seed": 1,
    "verbosity": 1,
}
num_boost_round = 30
output_file = "model.bin"

df = pd.read_csv("data/online_shoppers_intention.csv")
df.columns = df.columns.str.lower()

for col in ["month", "visitortype"]:
    df[col] = df[col].str.lower()

df["weekend"] = df["weekend"].astype(int)
df["revenue"] = df["revenue"].astype(int)

id_columns = ["operatingsystems", "browser", "region", "traffictype"]
for col in id_columns:
    df[col] = df[col].astype(str)

numerical = [
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "productrelated",
    "productrelated_duration",
    "bouncerates",
    "exitrates",
    "pagevalues",
    "specialday",
]

categorical = [
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
    "weekend",
]

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.revenue.values
y_test = df_test.revenue.values

del df_full_train["revenue"]
del df_test["revenue"]

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train[categorical + numerical].to_dict(orient="records")
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test[categorical + numerical].to_dict(orient="records")
X_test = dv.transform(dicts_test)

features = list(dv.get_feature_names_out())
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

print("Training the final model...")
model = xgb.train(xgb_params, dfulltrain, num_boost_round=100)

y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print(f"Final Test ROC AUC: {auc:.4f}")

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"The model is saved to {output_file}")
