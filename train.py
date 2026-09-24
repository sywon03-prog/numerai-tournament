"""
train.py — Numerai 모델 학습 (모델을 바꿀 때만 실행, 매일 돌리지 않음)

전체 흐름
  1. 데이터 다운로드      numerapi로 train / validation / features.json 받기
  2. 학습 데이터 로드     피처셋(medium)만 골라서 메모리에 올리기
  3. 모델 학습            LightGBM 회귀 모델
  4. 검증                 validation 데이터에서 era별 CORR 평균/표준편차
  5. 저장                 model.pkl (모델 + 피처 목록을 한 묶음으로)

실행:  python train.py
       NUMERAI_DATA_DIR=경로 python train.py   ← 이미 받아둔 데이터 폴더를 재사용할 때
"""

import json
import os
import pickle
import platform
import time
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numerai_tools.scoring import numerai_corr
from numerapi import NumerAPI

# ──────────────────────────────────────────────────────────────────────────────
# 설정
#   여기 값들이 곧 "이 모델의 정체"다. MODEL_CARD.md에 적는 내용과 일치해야 한다.
# ──────────────────────────────────────────────────────────────────────────────
DATA_VERSION = "v5.3"     # Numerai 데이터셋 버전. 새 버전은 napi.list_datasets()로 확인
FEATURE_SET = "medium"    # small(42) / medium(780) / all(3555). 전체는 메모리·시간 부담이 큼
TARGET = "target"         # 기본 타겟 (60일 후 수익률 기준). 다른 타겟들은 앙상블 단계에서 다룸

# 학습 데이터 마지막 era 이후 이만큼의 validation era는 평가에서 제외한다.
# 이유: 타겟이 "60 거래일(≈12주) 뒤"를 보기 때문에, 학습 마지막 주의 정답이
#       그다음 12주의 가격 움직임을 이미 담고 있다. 그 구간을 평가에 쓰면 커닝이 된다.
EMBARGO_ERAS = 12

# LightGBM 파라미터 — 공식 예제 기본값 그대로. 튜닝은 나중 단계.
PARAMS = {
    "n_estimators": 2000,       # 트리 개수
    "learning_rate": 0.01,      # 트리 하나가 정답에 다가가는 보폭 (작을수록 천천히, 안정적으로)
    "max_depth": 5,             # 트리 깊이 제한
    "num_leaves": 2**5 - 1,     # 트리 하나의 최대 잎 개수 (depth 5 → 31)
    "colsample_bytree": 0.1,    # 트리마다 피처의 10%만 무작위로 씀 → 과적합 완화
    "verbose": -1,
}

# 이미 받아둔 데이터가 있으면 NUMERAI_DATA_DIR로 그 폴더를 가리킨다 (기본은 ./data)
DATA_DIR = Path(os.environ.get("NUMERAI_DATA_DIR", "data")) / DATA_VERSION
MODEL_PATH = Path("model.pkl")
METRICS_PATH = Path("metrics.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 데이터 다운로드
#   데이터 다운로드는 API 키가 필요 없다(공개). 이미 같은 크기의 파일이 있으면 건너뛴다.
# ──────────────────────────────────────────────────────────────────────────────
log("1/5 데이터 다운로드")
napi = NumerAPI()
DATA_DIR.mkdir(parents=True, exist_ok=True)
for name in ("features.json", "train.parquet", "validation.parquet"):
    napi.download_dataset(f"{DATA_VERSION}/{name}", dest_path=str(DATA_DIR / name))

# features.json에는 피처셋별 피처 이름 목록이 들어 있다.
# 이 리스트의 "순서"까지 그대로 써야 한다 — 학습과 예측이 같은 순서로 컬럼을 넘겨야 하므로.
feature_metadata = json.load(open(DATA_DIR / "features.json"))
FEATURES = feature_metadata["feature_sets"][FEATURE_SET]
log(f"   피처셋 '{FEATURE_SET}': {len(FEATURES)}개")


# ──────────────────────────────────────────────────────────────────────────────
# 2. 학습 데이터 로드
#   parquet은 컬럼 단위로 읽을 수 있어서, 3555개 피처 중 필요한 것만 읽으면 메모리를 아낀다.
#   era = 1주일 단위의 시간 묶음. 같은 era의 행들은 "같은 날짜의 여러 종목"이다.
# ──────────────────────────────────────────────────────────────────────────────
log("2/5 학습 데이터 로드")
train = pd.read_parquet(DATA_DIR / "train.parquet", columns=["era", TARGET] + FEATURES)

# 타겟이 비어 있는 행은 정답이 없으니 학습에 쓸 수 없다.
n_before = len(train)
train = train.dropna(subset=[TARGET])
log(f"   {len(train):,}행 × {len(FEATURES)}피처, era {train['era'].min()}~{train['era'].max()}"
    f" (타겟 없는 {n_before - len(train):,}행 제외)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. 모델 학습
#   LightGBM 회귀: 작은 결정트리를 2000개 이어 붙여, 앞 트리의 오차를 뒤 트리가 조금씩 줄인다.
#   피처값이 비어 있는(NaN) 칸은 LightGBM이 알아서 처리한다.
# ──────────────────────────────────────────────────────────────────────────────
log("3/5 모델 학습 (몇 분 걸림)")
t0 = time.time()
model = lgb.LGBMRegressor(**PARAMS)
model.fit(train[FEATURES], train[TARGET])
train_seconds = time.time() - t0
log(f"   학습 완료: {train_seconds/60:.1f}분")

last_train_era = int(train["era"].max())
del train  # validation을 올릴 자리를 비운다


# ──────────────────────────────────────────────────────────────────────────────
# 4. 검증
#   validation은 학습에 쓰지 않은 미래 구간이다. 여기서의 성적이 "실전에서 어느 정도 나올지"의
#   가장 정직한 추정치다.
#
#   평가는 era 단위로 한다. Numerai 채점도 era(=라운드) 하나씩 CORR을 매기기 때문에,
#   전체를 뭉쳐 하나의 상관계수를 내면 실제 채점과 다른 숫자가 나온다.
#
#   CORR(numerai_corr): 예측의 순위와 실제 타겟의 상관. 0.02~0.03이면 준수, 음수면 돈을 잃는 주.
#   Sharpe = 평균 / 표준편차. "얼마나 꾸준히" 버는지를 본다. 평균만 높고 들쭉날쭉하면 낮게 나온다.
# ──────────────────────────────────────────────────────────────────────────────
log("4/5 검증")
# 엠바고: 학습 마지막 era 직후 12개 era는 제외 (위 EMBARGO_ERAS 설명 참고)
embargo = [str(last_train_era + i).zfill(4) for i in range(1, EMBARGO_ERAS + 1)]
validation = pd.read_parquet(
    DATA_DIR / "validation.parquet",
    columns=["era", "data_type", TARGET] + FEATURES,
    filters=[("data_type", "==", "validation"), ("era", "not in", embargo)],
).dropna(subset=[TARGET])
log(f"   {len(validation):,}행, era {validation['era'].min()}~{validation['era'].max()}"
    f" (엠바고 {embargo[0]}~{embargo[-1]} 제외)")

# era 하나씩 예측 → 채점. 한 번에 다 예측하면 메모리를 몇 배로 먹는다.
per_era_corr = {}
for era, era_df in validation.groupby("era"):
    preds = pd.DataFrame({"prediction": model.predict(era_df[FEATURES])}, index=era_df.index)
    per_era_corr[era] = numerai_corr(preds, era_df[TARGET])["prediction"]
per_era_corr = pd.Series(per_era_corr).sort_index()

metrics = {
    "corr_mean": float(per_era_corr.mean()),
    "corr_std": float(per_era_corr.std()),
    "corr_sharpe": float(per_era_corr.mean() / per_era_corr.std()),
    "corr_positive_ratio": float((per_era_corr > 0).mean()),  # 플러스인 era 비율
    "n_validation_eras": int(len(per_era_corr)),
}
log(f"   CORR 평균 {metrics['corr_mean']:.4f} | 표준편차 {metrics['corr_std']:.4f}"
    f" | Sharpe {metrics['corr_sharpe']:.2f} | 양수 era {metrics['corr_positive_ratio']:.0%}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. 저장
#   모델만 저장하지 않고 "피처 목록"을 같이 묶는다.
#   submit.py가 피처 이름·순서를 따로 알 필요가 없어지고, 학습과 예측이 어긋날 방법이 사라진다.
#   (이 프로젝트 초기에 실제로 겪은 버그: 학습은 372개 피처, 예측은 3555개를 넘겨서 에러)
# ──────────────────────────────────────────────────────────────────────────────
log("5/5 저장")
bundle = {
    "model": model,
    "features": FEATURES,
    "meta": {
        "trained_on": date.today().isoformat(),
        "data_version": DATA_VERSION,
        "feature_set": FEATURE_SET,
        "target": TARGET,
        "params": PARAMS,
        "train_eras": f"0001-{last_train_era:04d}",
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
    },
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

# 검증 결과도 파일로 남긴다 → MODEL_CARD.md 작성과 모델 간 비교에 사용
metrics.update(bundle["meta"])
metrics["train_minutes"] = round(train_seconds / 60, 1)
metrics["per_era_corr"] = {k: round(float(v), 5) for k, v in per_era_corr.items()}
json.dump(metrics, open(METRICS_PATH, "w"), indent=2, ensure_ascii=False)

size_mb = MODEL_PATH.stat().st_size / 1_000_000
log(f"   {MODEL_PATH} ({size_mb:.1f} MB), {METRICS_PATH}")
if size_mb > 100:
    log("   ⚠️  100MB 초과 — GitHub는 100MB 넘는 파일을 거부한다. Git LFS 또는 트리 수 축소 필요")
