# FC Online 플레이스타일 분석 프로젝트

FC Online Open API 경기 데이터를 활용하여 유저의 경기 행동 패턴을 분석하고,  
K-Means 군집화를 통해 플레이스타일을 분류하는 프로젝트입니다.

신규 유저의 최근 경기 데이터를 이용하여 기존에 학습된 K-Means 모델로 플레이스타일을 예측할 수 있습니다.

---

## 1. 주요 기능

- FC Online 경기 데이터 수집
- 경기 원본 데이터 CSV 관리
- 정상 종료 경기 기반 전처리
- 유저별 최근 70경기 행동 패턴 집계
- 플레이스타일 분석용 행동 Feature 14개 생성
- StandardScaler를 이용한 Feature 표준화
- K-Means 기반 플레이스타일 군집화
- 신규 유저 최근 30경기 기반 플레이스타일 예측
- Silhouette Score 및 ARI를 이용한 K-Means 실험

---

## 2. 전체 데이터 흐름

### 모델 학습

```text
match_data/match_team_data.csv
        ↓
machine_learning/feature_pipeline.py
        ↓
정상 종료 경기만 사용
        ↓
유저별 최근 70경기 집계
        ↓
14개 행동 Feature 생성
        ↓
machine_learning/data/
user_behavior_features_70matches.csv
        ↓
machine_learning/save_style_model.py
        ↓
StandardScaler + K-Means 학습
        ↓
machine_learning/models/
```

### 신규 유저 플레이스타일 예측

```text
신규 유저 최근 30경기 원본 CSV
        ↓
machine_learning/predict_user_style.py
        ↓
14개 행동 Feature 생성
        ↓
저장된 StandardScaler.transform()
        ↓
저장된 K-Means.predict()
        ↓
플레이스타일 예측
```

---

## 3. 프로젝트 구조

```text
fc-online-project/
│
├── archive/
│   ├── make_ml_features.py
│   ├── make_style_features.py
│   ├── make_user_style_features.py
│   └── match_ml_features.csv
│
├── docs/
│
├── machine_learning/
│   │
│   ├── data/
│   │   └── user_behavior_features_70matches.csv
│   │
│   ├── experiments/
│   │   ├── results/
│   │   ├── check_low_activity_cluster.py
│   │   ├── first_kmeans.py
│   │   ├── kmeans_behavior_only.py
│   │   └── kmeans_stability_test.py
│   │
│   ├── models/
│   │   ├── style_scaler.pkl
│   │   ├── style_kmeans_k4.pkl
│   │   └── style_features.pkl
│   │
│   ├── outputs/
│   │
│   ├── analyze_clusters.py
│   ├── feature_pipeline.py
│   ├── predict_user_style.py
│   └── save_style_model.py
│
├── match_data/
│   ├── raw_data/
│   ├── json_to_csv.py
│   ├── match_player_data.csv
│   ├── match_team_data.csv
│   ├── merge_match_csvs.py
│   └── test_match_api.py
│
├── player_data/
│   └── crawl_player_stats.py
│
├── .env
├── .gitignore
├── CLAUDE.md
├── requirements.txt
├── snowball_collect.py
└── README.md
```

---

## 4. 주요 폴더 설명

| 폴더 | 역할 |
|---|---|
| `match_data/` | FC Online 경기 원본 데이터 및 수집/변환 코드 |
| `match_data/raw_data/` | API에서 수집한 원본 JSON 데이터 |
| `machine_learning/data/` | 머신러닝 학습에 사용하는 가공 데이터 |
| `machine_learning/models/` | 학습 완료된 StandardScaler 및 K-Means 모델 |
| `machine_learning/outputs/` | 최종 분석 및 신규 유저 예측 결과 |
| `machine_learning/experiments/` | K값, Silhouette, ARI 등 실험 코드 |
| `machine_learning/experiments/results/` | K-Means 실험 결과 CSV |
| `player_data/` | 선수 데이터 관련 코드 |
| `archive/` | 현재 파이프라인에서 사용하지 않는 과거 코드 및 데이터 |

---

## 5. 설치 및 환경 설정

프로젝트를 Clone한 뒤 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

FC Online Open API를 사용하는 경우 프로젝트 루트에 `.env` 파일을 생성합니다.

```env
NEXON_API_KEY=본인의_API_KEY
```

`.env` 파일과 실제 API Key는 GitHub에 업로드하지 않습니다.

---

## 6. 핵심 원본 경기 데이터

플레이스타일 분석에서 사용하는 핵심 원본 파일은 다음과 같습니다.

```text
match_data/match_team_data.csv
```

현재 데이터는 기존 `match_team_data.csv`와 동일한 54개 컬럼 구조를 기준으로 사용합니다.

원본 데이터에는 다음과 같은 정보가 포함됩니다.

```text
matchId
matchDate
matchEndType
ouid
nickname
division

possession
dribble

passTry
shortPassTry
longPassTry
bouncingLobPassTry
drivenGroundPassTry
throughPassTry
lobbedThroughPassTry

shootTotal
shootHeading
shootInPenalty

tackleTry
blockTry

...
```

54개 컬럼 전체를 K-Means에 사용하는 것은 아닙니다.

`feature_pipeline.py`에서 플레이스타일 분석에 필요한 원본 컬럼만 사용하여 최종 14개 행동 Feature를 생성합니다.

---

## 7. 새로운 학습용 매치데이터를 받은 경우

새로운 원본 경기 데이터를 학습에 반영하려면 먼저 최종 원본 데이터가 다음 위치에 있어야 합니다.

```text
match_data/match_team_data.csv
```

새로운 데이터 역시 기존 원본과 호환되는 54개 컬럼 구조를 사용합니다.

### Step 1. 행동 Feature 생성

프로젝트 루트에서 다음 명령어를 실행합니다.

```bash
python machine_learning/feature_pipeline.py
```

`feature_pipeline.py`는 다음 작업을 수행합니다.

```text
match_team_data.csv
        ↓
정상 종료 경기(matchEndType == 0)만 사용
        ↓
동일 유저 + 동일 경기 중복 제거
        ↓
70경기 이상 보유한 유저 선택
        ↓
유저별 최근 70경기 선택
        ↓
14개 행동 Feature 계산
```

생성되는 파일:

```text
machine_learning/data/user_behavior_features_70matches.csv
```

현재 데이터 기준 결과:

```text
183명 × 18컬럼
```

18개 컬럼의 구성은 다음과 같습니다.

```text
식별/확인용 컬럼 4개
+
행동 Feature 14개
```

식별/확인용 컬럼:

```text
ouid
nickname
division
match_count
```

---

### Step 2. K-Means 모델 다시 학습

새로운 학습 데이터를 모델에 반영하려면 다음 명령어를 실행합니다.

```bash
python machine_learning/save_style_model.py
```

생성되는 모델 파일:

```text
machine_learning/models/
├── style_scaler.pkl
├── style_kmeans_k4.pkl
└── style_features.pkl
```

전체 흐름:

```text
match_team_data.csv
        ↓
feature_pipeline.py
        ↓
user_behavior_features_70matches.csv
        ↓
save_style_model.py
        ↓
Scaler + K-Means 모델 갱신
```

---

## 8. 플레이스타일 행동 Feature 14개

최종 K-Means 모델은 다음 14개의 행동 Feature를 사용합니다.

| Feature | 설명 |
|---|---|
| `possession` | 최근 경기 평균 점유 관련 지표 |
| `dribble` | 최근 경기 평균 드리블 관련 지표 |
| `pass_per_match` | 경기당 평균 패스 시도 |
| `short_pass_rate` | 전체 패스 중 짧은 패스 시도 비율 |
| `long_pass_rate` | 전체 패스 중 긴 패스 시도 비율 |
| `bouncing_lob_pass_rate` | 전체 패스 중 바운싱 로빙 패스 시도 비율 |
| `driven_ground_pass_rate` | 전체 패스 중 드리븐 그라운드 패스 시도 비율 |
| `through_pass_rate` | 전체 패스 중 스루패스 시도 비율 |
| `lobbed_through_pass_rate` | 전체 패스 중 로빙 스루패스 시도 비율 |
| `shoot_per_match` | 경기당 평균 슈팅 수 |
| `inside_penalty_rate` | 전체 슈팅 중 페널티박스 안 슈팅 비율 |
| `heading_shoot_rate` | 전체 슈팅 중 헤딩 슈팅 비율 |
| `tackle_per_match` | 경기당 평균 태클 시도 |
| `block_per_match` | 경기당 평균 블록 시도 |

플레이스타일 자체의 행동 패턴을 분석하기 위해 성공 여부보다는 행동의 선택과 빈도를 중심으로 Feature를 구성했습니다.

따라서 다음과 같은 성공률 중심 Feature는 최종 K-Means 입력에서 제외했습니다.

```text
pass_success_rate
effective_shoot_rate
tackle_success_rate
block_success_rate
```

이러한 Feature는 플레이스타일뿐 아니라 유저의 숙련도나 경기 성과의 영향을 받을 수 있기 때문에 행동 중심 군집화에서는 제외했습니다.

---

## 9. 행동 Feature 계산 방식

14개 행동 Feature 중 일부는 원본 CSV에 그대로 존재하지 않고 여러 경기를 집계하여 계산합니다.

예:

```text
pass_per_match
= 전체 passTry 합계 / 사용 경기 수
```

```text
short_pass_rate
= 전체 shortPassTry 합계 / 전체 passTry 합계
```

```text
long_pass_rate
= 전체 longPassTry 합계 / 전체 passTry 합계
```

```text
shoot_per_match
= 전체 shootTotal 합계 / 사용 경기 수
```

```text
inside_penalty_rate
= 전체 shootInPenalty 합계 / 전체 shootTotal 합계
```

```text
heading_shoot_rate
= 전체 shootHeading 합계 / 전체 shootTotal 합계
```

```text
tackle_per_match
= 전체 tackleTry 합계 / 사용 경기 수
```

```text
block_per_match
= 전체 blockTry 합계 / 사용 경기 수
```

`possession`과 `dribble`은 최근 경기의 평균값을 사용합니다.

학습 데이터와 신규 유저 데이터는 모두 동일한 Feature 계산 방식을 사용합니다.

---

## 10. K-Means 모델

현재 플레이스타일 모델 설정:

```text
Feature 수   : 14
K            : 4
random_state : 42
n_init       : 10
```

학습 과정:

```text
14개 행동 Feature
        ↓
StandardScaler.fit_transform()
        ↓
K-Means.fit()
        ↓
4개 Cluster
```

Feature마다 값의 크기와 단위가 다르기 때문에 K-Means 학습 전에 `StandardScaler`를 이용하여 표준화합니다.

---

## 11. K값 선정 및 안정성 실험

K값을 결정하기 위해 K=2 ~ K=8에 대한 Silhouette Score 비교를 수행했습니다.

대표적인 결과:

```text
K=2 : 약 0.148
K=3 : 약 0.126
K=4 : 약 0.124
K=5 : 약 0.125
```

Silhouette Score 기준에서는 K=2가 가장 높은 값을 보였습니다.

추가로 여러 random state에서 K-Means를 반복 실행하고 ARI(Adjusted Rand Index)를 이용하여 군집 결과의 안정성을 확인했습니다.

K=2는 매우 높은 안정성을 보였고, K=4는 K=2보다 낮지만 플레이스타일을 보다 세분화하여 제공할 수 있는 후보로 판단했습니다.

따라서 현재 프로젝트에서는 다음과 같이 해석합니다.

```text
K=2
→ 통계적 기준에서 강한 대분류 후보

K=4
→ 플레이스타일을 세분화하기 위한 서비스용 모델
```

최종 신규 유저 플레이스타일 예측에서는 K=4 모델을 사용합니다.

관련 실험 코드는 다음 위치에 있습니다.

```text
machine_learning/experiments/
```

---

## 12. 현재 K=4 플레이스타일 해석

현재 각 Cluster는 행동 Feature 평균 및 Z-score 중심값을 기준으로 다음과 같이 해석하고 있습니다.

| Cluster | 플레이스타일 |
|---:|---|
| 0 | 빠른 전개·특수 패스형 |
| 1 | 직선·롱패스 침투형 |
| 2 | 공격 전개·수비 개입형 |
| 3 | 점유·짧은 패스 연계형 |

위 플레이스타일 이름은 K-Means가 직접 학습한 정답 Label이 아닙니다.

K-Means는 비지도학습이므로 실제로는 다음과 같은 Cluster 번호만 생성합니다.

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
```

각 Cluster의 Feature 특성을 분석한 후 사람이 이해하기 쉽도록 플레이스타일 이름을 부여한 것입니다.

---

## 13. 저장된 모델 파일

모델 파일은 다음 위치에 저장됩니다.

```text
machine_learning/models/
```

### style_scaler.pkl

```text
machine_learning/models/style_scaler.pkl
```

학습 데이터의 14개 행동 Feature를 기준으로 학습된 `StandardScaler`입니다.

신규 유저의 Feature를 기존 학습 데이터와 동일한 기준으로 표준화하기 위해 사용합니다.

---

### style_kmeans_k4.pkl

```text
machine_learning/models/style_kmeans_k4.pkl
```

K=4로 학습한 최종 플레이스타일 K-Means 모델입니다.

---

### style_features.pkl

```text
machine_learning/models/style_features.pkl
```

K-Means 모델이 사용하는 14개 Feature의 이름과 순서를 저장합니다.

신규 유저 예측 시 학습 당시와 동일한 Feature 순서를 유지하기 위해 사용합니다.

---

## 14. 신규 유저 최근 30경기 플레이스타일 예측

신규 유저의 플레이스타일을 분석할 때는 K-Means 모델을 다시 학습하지 않습니다.

기존에 저장된 모델을 이용하여 예측만 수행합니다.

### Step 1. 신규 유저 데이터 준비

신규 유저의 최근 경기 데이터를 기존 `match_team_data.csv`와 동일한 54개 컬럼 구조의 CSV로 준비합니다.

예:

```text
match_data/new_user_30matches.csv
```

신규 유저 분석에서는 정상 종료된 최근 30경기를 사용합니다.

---

### Step 2. INPUT_FILE 확인

`machine_learning/predict_user_style.py`에서 신규 CSV 경로를 확인합니다.

예:

```python
INPUT_FILE = "match_data/new_user_30matches.csv"
```

---

### Step 3. 플레이스타일 예측 실행

```bash
python machine_learning/predict_user_style.py
```

내부 처리 과정:

```text
신규 유저 30경기 원본 데이터
        ↓
build_behavior_features()
        ↓
14개 행동 Feature 생성
        ↓
style_features.pkl 기준 Feature 선택
        ↓
style_scaler.pkl
        ↓
transform()
        ↓
style_kmeans_k4.pkl
        ↓
predict()
        ↓
Cluster 0 ~ 3
        ↓
플레이스타일 출력
```

---

## 15. 신규 유저 예측 시 주의사항

신규 유저를 예측할 때는 모델을 다시 학습하지 않습니다.

따라서 다음 코드는 신규 유저 예측에서 사용하지 않습니다.

```python
scaler.fit()
scaler.fit_transform()

kmeans.fit()
kmeans.fit_predict()
```

대신 기존에 저장된 모델에서 다음 함수만 사용합니다.

```python
scaler.transform()
kmeans.predict()
```

즉 신규 유저가 들어올 때마다 새로운 군집을 만드는 것이 아니라, 기존에 만들어진 4개 Cluster 중 어느 Cluster와 가장 가까운지를 판단합니다.

---

## 16. 학습용 70경기와 예측용 30경기

모델 학습에서는 유저의 플레이스타일을 보다 안정적으로 표현하기 위해 유저별 최근 70경기를 사용합니다.

```text
학습 데이터
유저 1명 = 최근 70경기
```

신규 유저 예측에서는 데이터 수집 부담을 줄이기 위해 최근 30경기를 사용합니다.

```text
신규 유저 예측
유저 1명 = 최근 30경기
```

사용 경기 수는 다르지만 행동 Feature를 계산하는 공식과 Feature 순서는 동일하게 유지합니다.

---

## 17. 예측 결과

신규 유저 예측 결과는 다음 위치에 저장됩니다.

```text
machine_learning/outputs/new_user_style_prediction.csv
```

결과에는 다음 정보가 포함됩니다.

```text
ouid
nickname
division
match_count

14개 행동 Feature

cluster
style_name
```

예측 과정에서는 신규 유저와 각 Cluster 중심까지의 거리도 확인할 수 있습니다.

Cluster 중심까지의 거리가 작을수록 해당 Cluster의 중심적인 플레이 패턴과 더 가깝다는 의미입니다.

단, Cluster 중심까지의 거리는 예측 확률이나 정확도를 의미하지 않습니다.

---

## 18. Experiments 폴더

```text
machine_learning/experiments/
```

최종 예측 과정에서 직접 사용하지 않는 K-Means 실험 코드를 보관합니다.

주요 내용:

```text
K값 비교
Silhouette Score 분석
ARI 안정성 분석
초기 K-Means 실험
저활동 Cluster 분석
```

실험 결과 CSV는 다음 위치에 보관합니다.

```text
machine_learning/experiments/results/
```

---

## 19. Archive 폴더

```text
archive/
```

과거 데이터 전처리 방식 및 이전 Feature 생성 과정에서 사용했던 코드와 데이터를 보관합니다.

현재 플레이스타일 K-Means 파이프라인에서는 사용하지 않습니다.

현재 플레이스타일 분석의 핵심 코드는 다음 3개입니다.

```text
machine_learning/feature_pipeline.py
machine_learning/save_style_model.py
machine_learning/predict_user_style.py
```

---

## 20. 실행 순서 요약

### 새로운 학습용 경기 데이터를 사용할 때

```text
1. match_data/match_team_data.csv 준비

        ↓

2. 행동 Feature 생성

python machine_learning/feature_pipeline.py

        ↓

3. 생성 결과 확인

machine_learning/data/
user_behavior_features_70matches.csv

        ↓

4. 모델 학습 및 저장

python machine_learning/save_style_model.py

        ↓

5. 모델 확인

machine_learning/models/
```

### 신규 유저의 플레이스타일만 예측할 때

```text
1. 신규 유저 최근 30경기 CSV 준비

        ↓

2. predict_user_style.py의 INPUT_FILE 설정

        ↓

3. 플레이스타일 예측 실행

python machine_learning/predict_user_style.py

        ↓

4. 14개 행동 Feature 자동 생성

        ↓

5. 기존 StandardScaler.transform()

        ↓

6. 기존 K-Means.predict()

        ↓

7. Cluster 및 플레이스타일 확인

        ↓

8. 결과 확인

machine_learning/outputs/
new_user_style_prediction.csv
```

---

## 21. 핵심 파일만 빠르게 보기

처음 프로젝트를 확인하는 경우 아래 파일들을 우선 확인하면 됩니다.

```text
match_data/match_team_data.csv
        │
        │ 원본 경기 데이터
        ↓
machine_learning/feature_pipeline.py
        │
        │ 행동 Feature 생성
        ↓
machine_learning/data/
user_behavior_features_70matches.csv
        │
        │ K-Means 학습 데이터
        ↓
machine_learning/save_style_model.py
        │
        │ 모델 학습 및 저장
        ↓
machine_learning/models/
        │
        │ 저장된 Scaler / K-Means
        ↓
machine_learning/predict_user_style.py
        │
        │ 신규 유저 플레이스타일 예측
        ↓
machine_learning/outputs/
```

플레이스타일 K-Means와 관련된 핵심 Python 파일은 다음 세 개입니다.

```text
feature_pipeline.py
save_style_model.py
predict_user_style.py
```