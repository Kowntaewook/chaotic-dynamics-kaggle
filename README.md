# Chaotic Dynamics Kaggle Baseline

Kaggle의 Multi-System Chaotic Dynamics Dataset을 활용하여 3차원 상태 벡터의 다음 시점을 예측하는 GRU 기반 시계열 회귀 모델입니다.

CNN은 사용하지 않았습니다. 본 프로젝트에서 사용하는 데이터는 이미지가 아니라 시간 순서에 따라 변화하는 벡터 데이터이므로, 순차 패턴을 학습할 수 있는 GRU 모델을 사용했습니다.

## Dataset

- Kaggle dataset: namandixit07/multi-system-chaotic-dynamics-dataset
- 사용 파일: embeddings_umap3d.csv
- 사용 feature: UMAP_1, UMAP_2, UMAP_3

UMAP_1, UMAP_2, UMAP_3를 하나의 3차원 상태 벡터로 보고, 과거 32개 시점의 상태를 입력받아 다음 시점의 상태를 예측하도록 구성했습니다.

## Model

- 입력: 과거 32개 시점의 3차원 상태 벡터
- 출력: 다음 시점의 3차원 상태 벡터
- 모델: GRU
- 손실 함수: MSE Loss
- Optimizer: AdamW
- 평가 지표: RMSE

```text
Input  shape: (batch, 32, 3)
Output shape: (batch, 3)
Install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Run

데이터 로딩 테스트:

.\.venv\Scripts\python.exe chaotic_forecast.py --no-train --file-glob embeddings_umap3d.csv --state-cols UMAP_1,UMAP_2,UMAP_3 --seq-len 32 --horizon 1

학습 실행:

.\.venv\Scripts\python.exe chaotic_forecast.py --file-glob embeddings_umap3d.csv --state-cols UMAP_1,UMAP_2,UMAP_3 --seq-len 32 --horizon 1 --epochs 10 --max-windows 30000

KaggleHub cache 직접 지정:

.\.venv\Scripts\python.exe chaotic_forecast.py --data-dir "C:\Users\Administrator\.cache\kagglehub\datasets\namandixit07\multi-system-chaotic-dynamics-dataset\versions\19" --file-glob embeddings_umap3d.csv --state-cols UMAP_1,UMAP_2,UMAP_3 --seq-len 32 --horizon 1 --epochs 10 --max-windows 30000
Training Result
train windows: (3968, 32, 3)
test windows : (1000, 32, 3)
epoch 010/10 mse=1.000442
overall RMSE: 4.077438
Outputs
FileDescription
eda_summary.csvfeature별 기초 통계량
trajectory_preview.png3차원 UMAP trajectory 시각화
training_loss.pngepoch별 training loss 그래프
forecast_preview.png실제값과 예측값 비교 그래프
predictions_head.csv예측 결과 일부
metrics.jsonRMSE 평가 결과
gru_forecaster.pt학습된 PyTorch 모델 checkpoint

outputs, .venv, 모델 checkpoint 파일은 .gitignore에 포함하여 GitHub에는 업로드하지 않습니다.

Summary

이 프로젝트는 chaotic dynamics 형태의 3차원 embedding 데이터를 시계열 문제로 해석하고, GRU 모델을 이용해 다음 상태를 예측하는 baseline입니다. CNN 없이 순차 모델만 사용했으며, 학습 결과와 평가 지표를 outputs/metrics.json으로 확인할 수 있습니다.
