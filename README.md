# numerai-tournament

Numerai 토너먼트용 LightGBM 모델.

```
pip install -r requirements.txt
cp .env.example .env

python train.py            # 데이터 받고 학습. model.pkl, metrics.json 생성
python submit.py           # live 예측해서 제출. --dry-run 붙이면 제출은 안 함
```

데이터 폴더를 따로 두고 싶으면 `NUMERAI_DATA_DIR`로 지정.

제출은 GitHub Actions에서 화~토 14:00 UTC에 돌아감 (`.github/workflows/numerai.yml`). 키는 repository secrets에.
