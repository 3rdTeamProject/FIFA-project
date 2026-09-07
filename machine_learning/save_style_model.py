import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# ============================================================
# 1. 설정
# ============================================================

INPUT_FILE = (
    "machine_learning/data/"
    "user_behavior_features_70matches.csv"
)

MODEL_DIR = "machine_learning/models"
OUTPUT_DIR = "machine_learning/outputs"

K = 4
RANDOM_STATE = 42
N_INIT = 10


# ============================================================
# 2. K-Means에 사용할 14개 Feature
# ============================================================

FEATURES = [
    "possession",
    "dribble",
    "pass_per_match",
    "short_pass_rate",
    "long_pass_rate",
    "bouncing_lob_pass_rate",
    "driven_ground_pass_rate",
    "through_pass_rate",
    "lobbed_through_pass_rate",
    "shoot_per_match",
    "inside_penalty_rate",
    "heading_shoot_rate",
    "tackle_per_match",
    "block_per_match",
]


# ============================================================
# 3. 저장 폴더 생성
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 4. 데이터 불러오기
# ============================================================

print("=" * 70)
print("K=4 플레이스타일 모델 저장")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\n데이터 크기:",
    df.shape
)


# ============================================================
# 5. Feature 존재 여부 확인
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    print()
    print("[오류] 필요한 Feature가 없습니다.")

    for feature in missing_features:
        print(
            "-",
            feature
        )

    raise ValueError(
        "user_style_features_70matches.csv의 컬럼을 확인해주세요."
    )


# ============================================================
# 6. K-Means 입력 데이터 준비
# ============================================================

X = df[
    FEATURES
].copy()


print(
    "\n사용 Feature 수:",
    len(FEATURES)
)

print(
    "데이터 행 수:",
    len(X)
)

print(
    "결측치 개수:",
    X.isna().sum().sum()
)


if X.isna().sum().sum() > 0:

    raise ValueError(
        "K-Means 입력 데이터에 결측치가 있습니다."
    )


# ============================================================
# 7. StandardScaler 학습
# ============================================================
#
# 중요:
# 여기서는 183명 학습 데이터 기준으로
# 평균/표준편차를 계산한다.
#
# 나중에 신규 유저는 fit_transform 하지 않고
# 이 scaler를 불러와 transform만 한다.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

print(
    "\nStandardScaler 학습 완료"
)


# ============================================================
# 8. K=4 K-Means 학습
# ============================================================

kmeans = KMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

labels = kmeans.fit_predict(
    X_scaled
)

print(
    f"K-Means 학습 완료: K={K}"
)


# ============================================================
# 9. 실루엣 점수 확인
# ============================================================

score = silhouette_score(
    X_scaled,
    labels
)

print(
    f"Silhouette Score: {score:.4f}"
)


# ============================================================
# 10. 군집 결과 생성
# ============================================================

result_df = df.copy()

result_df[
    "cluster"
] = labels


# ============================================================
# 11. 군집별 유저 수 확인
# ============================================================

counts = (
    result_df[
        "cluster"
    ]
    .value_counts()
    .sort_index()
)

print()
print("=" * 70)
print("군집별 유저 수")
print("=" * 70)

for cluster_id, count in counts.items():

    ratio = (
        count
        / len(result_df)
        * 100
    )

    print(
        f"Cluster {cluster_id}: "
        f"{count}명 "
        f"({ratio:.2f}%)"
    )


# ============================================================
# 12. 군집별 원본 Feature 평균
# ============================================================

raw_means = (
    result_df
    .groupby(
        "cluster"
    )[FEATURES]
    .mean()
)


# ============================================================
# 13. 군집별 Z-score 중심
# ============================================================

centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=FEATURES
)

centers.index.name = "cluster"


print()
print("=" * 70)
print("군집별 Z-score 중심")
print("=" * 70)

print(
    centers.round(2)
)


# ============================================================
# 14. 학습된 모델 저장
# ============================================================

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "style_scaler.pkl"
)

KMEANS_FILE = os.path.join(
    MODEL_DIR,
    "style_kmeans_k4.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "style_features.pkl"
)


joblib.dump(
    scaler,
    SCALER_FILE
)

joblib.dump(
    kmeans,
    KMEANS_FILE
)

joblib.dump(
    FEATURES,
    FEATURE_FILE
)


# ============================================================
# 15. 분석 결과 CSV 저장
# ============================================================

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "user_style_k4_cluster_result.csv"
)

CENTER_FILE = os.path.join(
    OUTPUT_DIR,
    "user_style_k4_zscore_centers.csv"
)

RAW_MEAN_FILE = os.path.join(
    OUTPUT_DIR,
    "user_style_k4_raw_means.csv"
)


result_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)

centers.to_csv(
    CENTER_FILE,
    encoding="utf-8-sig"
)

raw_means.to_csv(
    RAW_MEAN_FILE,
    encoding="utf-8-sig"
)


# ============================================================
# 16. 최종 출력
# ============================================================

print()
print("=" * 70)
print("모델 저장 완료")
print("=" * 70)

print(
    "\nScaler 저장:",
    SCALER_FILE
)

print(
    "K-Means 저장:",
    KMEANS_FILE
)

print(
    "Feature 목록 저장:",
    FEATURE_FILE
)

print()
print(
    "Cluster 결과 저장:",
    RESULT_FILE
)

print(
    "Z-score 중심 저장:",
    CENTER_FILE
)

print(
    "원본 Feature 평균 저장:",
    RAW_MEAN_FILE
)

print()
print("=" * 70)