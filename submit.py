import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

base = Path(__file__).resolve().parent
load_dotenv(base / ".env")

with open(base / "model.pkl", "rb") as f:
    bundle = pickle.load(f)
model = bundle["model"]
features = bundle["features"]
version = bundle["meta"]["data_version"]

napi = NumerAPI(os.getenv("NUMERAI_PUBLIC_ID"), os.getenv("NUMERAI_SECRET_KEY"))
round_num = napi.get_current_round()

live_path = base / "data" / version / "live.parquet"
live_path.parent.mkdir(parents=True, exist_ok=True)
live_path.unlink(missing_ok=True)  # numerapi skips the download if the size matches
napi.download_dataset(f"{version}/live.parquet", str(live_path))

live = pd.read_parquet(live_path, columns=features)
preds = model.predict(live[features])
submission = pd.DataFrame({"id": live.index, "prediction": preds})
print(f"round {round_num}: {len(submission)} rows")

if args.dry_run:
    print(submission.head())
else:
    sid = napi.upload_predictions(df=submission, model_id=os.environ["NUMERAI_MODEL_ID"])
    print(f"submitted {sid}")
