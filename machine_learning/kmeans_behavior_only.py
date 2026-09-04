import os

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 설정
# ============================================================

INPUT_FILE = "match_data/match_ml_features.csv"

RESULT_DIR = "machine_learning"

RANDOM_STATE = 42

N_INIT = 10


# ============================================================
# 1. 데이터 불러오기
# ============================================================

print("=" * 70)
print("K-Means 행동 중심 Feature 실험")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\n데이터 크기:",
    df.shape
)


# ============================================================
# 2. K-Means에 사용할 행동 중심 Feature
# ============================================================
#
# 기존 18개 Feature 중
#
# pass_success_rate
# effective_shoot_rate
# tackle_success_rate
# block_success_rate
#
# 성공률 4개를 제외함.
#
# 원본 CSV에서는 삭제하지 않고
# K-Means 입력에서만 제외.
# ============================================================

features = [
    "possession",
    "dribble",
    "pass_per_possession",
    "short_pass_rate",
    "long_pass_rate",
    "bouncing_lob_pass_rate",
    "driven_ground_pass_rate",
    "through_pass_rate",
    "lobbed_through_pass_rate",
    "shoot_per_possession",
    "inside_penalty_rate",
    "heading_shoot_rate",
    "tackle_per_possession",
    "block_per_possession",
]


print(
    "\n사용 Feature 수:",
    len(features)
)

print(
    "\n사용 Feature:"
)

for feature in features:
    print(
        f" - {feature}"
    )


# ============================================================
# 3. Feature 데이터 추출
# ============================================================

X = df[
    features
].copy()


# ============================================================
# 4. 결측치 / 무한대 확인
# ============================================================

print(
    "\n결측치 개수:",
    X.isna().sum().sum()
)

print(
    "데이터 행 수:",
    len(X)
)


# ============================================================
# 5. 표준화
# ============================================================
#
# Feature마다 단위가 다르기 때문에
# 평균 0 / 표준편차 1 기준으로 맞춘다.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


print(
    "\n표준화 완료"
)


# ============================================================
# 6. K=2 ~ 8 실루엣 점수 비교
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "K=2 ~ 8 실루엣 점수"
)

print(
    "=" * 70
)


silhouette_results = []


for k in range(
    2,
    9
):

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_results.append(
        {
            "k": k,
            "silhouette_score": score
        }
    )

    print(
        f"K={k} "
        f"→ silhouette = {score:.4f}"
    )


# ============================================================
# 7. 실루엣 결과 CSV 저장
# ============================================================

df_silhouette = pd.DataFrame(
    silhouette_results
)

silhouette_file = os.path.join(
    RESULT_DIR,
    "behavior_silhouette_scores.csv"
)

df_silhouette.to_csv(
    silhouette_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. K=3 / K=4 상세 분석 함수
# ============================================================

def analyze_k(k):

    print(
        "\n" + "=" * 70
    )

    print(
        f"K={k} 상세 분석"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # K-Means 학습
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = kmeans.fit_predict(
        X_scaled
    )


    # --------------------------------------------------------
    # 원본 데이터에 cluster 번호 추가
    # --------------------------------------------------------

    result_df = df.copy()

    result_df[
        "cluster"
    ] = labels


    # --------------------------------------------------------
    # 군집별 개수
    # --------------------------------------------------------

    counts = (
        result_df[
            "cluster"
        ]
        .value_counts()
        .sort_index()
    )


    ratios = (
        counts
        / len(result_df)
        * 100
    )


    print(
        "\n군집별 데이터 개수"
    )


    for cluster_id in counts.index:

        print(
            f"Cluster {cluster_id}: "
            f"{counts[cluster_id]}개 "
            f"({ratios[cluster_id]:.2f}%)"
        )


    # --------------------------------------------------------
    # 군집별 원본 Feature 평균
    # --------------------------------------------------------

    raw_means = (
        result_df
        .groupby(
            "cluster"
        )[
            features
        ]
        .mean()
    )


    print(
        "\n군집별 원본 Feature 평균"
    )

    print(
        raw_means.round(4)
    )


    # --------------------------------------------------------
    # Z-score 기준 군집 중심
    # --------------------------------------------------------

    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=features
    )

    centers.index.name = "cluster"


    print(
        "\n군집별 Z-score 중심"
    )

    print(
        centers.round(2)
    )


    # --------------------------------------------------------
    # 각 군집에서 가장 높은/낮은 Feature
    # --------------------------------------------------------

    print(
        "\n군집별 특징 요약"
    )


    for cluster_id in centers.index:

        center = centers.loc[
            cluster_id
        ]

        top_high = center.sort_values(
            ascending=False
        ).head(5)

        top_low = center.sort_values(
            ascending=True
        ).head(5)


        print(
            f"\n[Cluster {cluster_id}]"
        )

        print(
            "높은 Feature TOP 5"
        )

        for feature_name, value in top_high.items():

            print(
                f"  + {feature_name}: "
                f"{value:.2f}"
            )


        print(
            "낮은 Feature TOP 5"
        )

        for feature_name, value in top_low.items():

            print(
                f"  - {feature_name}: "
                f"{value:.2f}"
            )


    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    result_file = os.path.join(
        RESULT_DIR,
        f"behavior_k{k}_cluster_result.csv"
    )

    result_df.to_csv(
        result_file,
        index=False,
        encoding="utf-8-sig"
    )


    center_file = os.path.join(
        RESULT_DIR,
        f"behavior_k{k}_zscore_centers.csv"
    )

    centers.to_csv(
        center_file,
        encoding="utf-8-sig"
    )


    raw_mean_file = os.path.join(
        RESULT_DIR,
        f"behavior_k{k}_raw_means.csv"
    )

    raw_means.to_csv(
        raw_mean_file,
        encoding="utf-8-sig"
    )


    print(
        f"\n저장 완료:"
    )

    print(
        f" - {result_file}"
    )

    print(
        f" - {center_file}"
    )

    print(
        f" - {raw_mean_file}"
    )


# ============================================================
# 9. K=3 / K=4 상세 분석 실행
# ============================================================

analyze_k(
    3
)

analyze_k(
    4
)


# ============================================================
# 10. 종료
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "행동 중심 Feature K-Means 실험 완료"
)

print(
    "=" * 70
)

print(
    "\n실루엣 결과:",
    silhouette_file
)