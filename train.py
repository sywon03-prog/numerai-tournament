import json
import os
import pickle
import platform
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from numerai_tools.scoring import numerai_corr
from numerapi import NumerAPI

DATA_VERSION = "v5.3"
FEATURE_SET = "medium"
TARGET = "target"
EMBARGO = 12  # target_60 overlaps the next ~12 eras after the last train era

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=31,
    colsample_bytree=0.1,
    verbose=-1,
)

data_dir = Path(os.environ.get("NUMERAI_DATA_DIR", "data")) / DATA_VERSION
data_dir.mkdir(parents=True, exist_ok=True)

napi = NumerAPI()
for name in ["features.json", "train.parquet", "validation.parquet"]:
    napi.download_dataset(f"{DATA_VERSION}/{name}", str(data_dir / name))

features = json.load(open(data_dir / "features.json"))["feature_sets"][FEATURE_SET]

train = pd.read_parquet(data_dir / "train.parquet", columns=["era", TARGET] + features)
train = train.dropna(subset=[TARGET])
print(f"train: {len(train)} rows, {len(features)} features")

t0 = time.time()
model = lgb.LGBMRegressor(**PARAMS)
model.fit(train[features], train[TARGET])
train_minutes = (time.time() - t0) / 60
print(f"fit: {train_minutes:.1f} min")

last_era = int(train["era"].max())
del train

embargo = [f"{last_era + i:04d}" for i in range(1, EMBARGO + 1)]
val = pd.read_parquet(
    data_dir / "validation.parquet",
    columns=["era", "data_type", TARGET] + features,
    filters=[("data_type", "==", "validation"), ("era", "not in", embargo)],
).dropna(subset=[TARGET])

corrs = {}
for era, df in val.groupby("era"):
    pred = pd.DataFrame({"prediction": model.predict(df[features])}, index=df.index)
    corrs[era] = numerai_corr(pred, df[TARGET])["prediction"]
corrs = pd.Series(corrs)
print(f"corr mean {corrs.mean():.4f}  std {corrs.std():.4f}  sharpe {corrs.mean() / corrs.std():.2f}")

meta = {
    "trained_on": date.today().isoformat(),
    "data_version": DATA_VERSION,
    "feature_set": FEATURE_SET,
    "target": TARGET,
    "params": PARAMS,
    "train_eras": f"0001-{last_era:04d}",
    "python": platform.python_version(),
    "lightgbm": lgb.__version__,
}

with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": features, "meta": meta}, f)

metrics = {
    "corr_mean": corrs.mean(),
    "corr_std": corrs.std(),
    "corr_sharpe": corrs.mean() / corrs.std(),
    "corr_positive_ratio": (corrs > 0).mean(),
    "n_validation_eras": len(corrs),
    **meta,
    "train_minutes": round(train_minutes, 1),
    "per_era_corr": corrs.round(5).to_dict(),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"model.pkl {os.path.getsize('model.pkl') / 1e6:.1f} MB")
