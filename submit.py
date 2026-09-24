"""
submit.py — 오늘 라운드 예측 제출 (매일 실행. 로컬에서는 사람이, GitHub Actions에서는 스케줄러가)

전체 흐름
  1. model.pkl 로드        train.py가 저장한 묶음 (모델 + 피처 목록)
  2. live 데이터 다운로드   오늘 라운드의 종목별 피처 (12MB)
  3. 예측                  피처 목록 순서 그대로 모델에 넣기
  4. 제출                  numerapi로 업로드

실행:  python submit.py            제출까지
       python submit.py --dry-run  예측까지만 하고 업로드는 하지 않음 (테스트용)

API 키와 model_id는 환경변수로만 받는다. 로컬은 .env 파일, GitHub Actions는 Secrets.
"""

import os
import pickle
import sys
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from numerapi import NumerAPI

DRY_RUN = "--dry-run" in sys.argv
BASE = Path(__file__).resolve().parent  # 어느 폴더에서 실행해도 이 파일 옆을 기준으로 잡는다

# .env가 있으면 읽어서 환경변수로 올린다. GitHub Actions에는 .env가 없고 Secrets가
# 환경변수로 직접 들어오므로, 그 경우 이 줄은 아무 일도 하지 않는다.
load_dotenv(BASE / ".env")
PUBLIC_ID = os.environ.get("NUMERAI_PUBLIC_ID")
SECRET_KEY = os.environ.get("NUMERAI_SECRET_KEY")
MODEL_ID = os.environ.get("NUMERAI_MODEL_ID")
if not DRY_RUN and not (PUBLIC_ID and SECRET_KEY and MODEL_ID):
    sys.exit("환경변수 NUMERAI_PUBLIC_ID / NUMERAI_SECRET_KEY / NUMERAI_MODEL_ID 가 필요합니다 (.env.example 참고)")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 모델 묶음 로드
#   피처 이름·순서를 여기서 꺼내 쓰므로, 이 파일은 "어떤 피처셋으로 학습했는지" 몰라도 된다.
#   train.py에서 FEATURE_SET을 바꿔 재학습하면 submit.py는 손댈 것 없이 그대로 따라간다.
# ──────────────────────────────────────────────────────────────────────────────
log("1/4 model.pkl 로드")
with open(BASE / "model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, features, meta = bundle["model"], bundle["features"], bundle["meta"]
DATA_VERSION = meta["data_version"]
log(f"   {meta['feature_set']} 피처셋 {len(features)}개, {meta['trained_on']} 학습")


# ──────────────────────────────────────────────────────────────────────────────
# 2. live 데이터 다운로드
#   live.parquet = "오늘 라운드의 종목들 × 피처". 정답(타겟)은 당연히 없다.
#   numerapi는 같은 크기의 파일이 이미 있으면 다운로드를 건너뛴다. 어제 파일이 우연히
#   같은 크기면 어제 데이터로 제출하게 되므로, 받기 전에 반드시 지운다.
# ──────────────────────────────────────────────────────────────────────────────
log("2/4 live 데이터 다운로드")
napi = NumerAPI(PUBLIC_ID, SECRET_KEY)
current_round = napi.get_current_round()
live_path = BASE / "data" / DATA_VERSION / "live.parquet"
live_path.parent.mkdir(parents=True, exist_ok=True)
live_path.unlink(missing_ok=True)
napi.download_dataset(f"{DATA_VERSION}/live.parquet", dest_path=str(live_path))
live = pd.read_parquet(live_path, columns=features)  # 필요한 컬럼만 읽는다
log(f"   라운드 {current_round}: {len(live):,}종목")


# ──────────────────────────────────────────────────────────────────────────────
# 3. 예측
#   live[features] — 학습 때와 같은 피처를 같은 순서로. 이 한 줄이 학습/예측 불일치를 막는다.
#   결과는 종목별 숫자 하나(상대 순위 점수). 절대값은 의미 없고 종목 간 크고 작음만 채점된다.
# ──────────────────────────────────────────────────────────────────────────────
log("3/4 예측")
preds = model.predict(live[features])
# Numerai 제출 형식: id 컬럼 + prediction 컬럼.
# numerapi가 DataFrame을 CSV로 바꿀 때 인덱스를 버리므로, id를 인덱스가 아닌 컬럼으로 빼둔다.
submission = pd.Series(preds, index=live.index, name="prediction").reset_index()
assert list(submission.columns) == ["id", "prediction"], submission.columns
assert submission["prediction"].notna().all(), "예측에 NaN이 있음"
log(f"   {len(submission):,}행, 예측값 범위 {preds.min():.3f} ~ {preds.max():.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. 제출
#   같은 라운드에 다시 제출하면 마지막 것으로 덮어써진다. 잘못 냈으면 다시 돌리면 된다.
# ──────────────────────────────────────────────────────────────────────────────
if DRY_RUN:
    log("4/4 [dry-run] 업로드 생략. 제출 미리보기:")
    print(submission.head(3).to_string(index=False))
    sys.exit(0)

log("4/4 제출")
submission_id = napi.upload_predictions(df=submission, model_id=MODEL_ID)
log(f"   완료 — 라운드 {current_round}, submission_id={submission_id}")
