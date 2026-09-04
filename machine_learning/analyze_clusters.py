import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ==================================================
# 1. 데이터 불러오기
# ==================================================

file_path = "match_data/match_ml_features.csv"

df = pd.read_csv(file_path)

print("=" * 80)
print("K-Means Cluster 특징 분석")
print("=" * 80)

print("\n전체 데이터 크기:", df.shape)


# ==================================================
# 2. K-Means에 사용할 18개 Feature
# ==================================================

features = [

    # 점유 / 개인 전개
    "possession",
    "dribble",

    # 패스
    "pass_per_possession",
    "pass_success_rate",
    "short_pass_rate",
    "long_pass_rate",
    "bouncing_lob_pass_rate",
    "driven_ground_pass_rate",
    "through_pass_rate",
    "lobbed_through_pass_rate",

    # 공격 / 슈팅
    "shoot_per_possession",
    "effective_shoot_rate",
    "inside_penalty_rate",
    "heading_shoot_rate",

    # 수비
    "tackle_per_possession",
    "tackle_success_rate",
    "block_per_possession",
    "block_success_rate"
]


X = df[features].copy()


# ==================================================
# 3. StandardScaler
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==================================================
# 4. 분석할 K 값
# ==================================================

candidate_k = [3, 4]


# ==================================================
# 5. K=3, K=4 분석
# ==================================================

for k in candidate_k:

    print("\n" + "=" * 80)
    print(f"K = {k} Cluster Z-score 분석")
    print("=" * 80)


    # --------------------------------------------------
    # K-Means 학습
    # --------------------------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)


    # --------------------------------------------------
    # Cluster별 데이터 개수와 비율
    # --------------------------------------------------

    cluster_counts = pd.Series(labels).value_counts().sort_index()

    cluster_ratio = (
        pd.Series(labels)
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )


    print("\n[Cluster별 개수 / 비율]")

    for cluster in range(k):

        count = cluster_counts[cluster]
        ratio = cluster_ratio[cluster]

        print(
            f"Cluster {cluster}: "
            f"{count}개 "
            f"({ratio:.2f}%)"
        )


    # ==================================================
    # 6. Cluster 중심값
    # ==================================================

    # model.cluster_centers_는
    # 이미 StandardScaler가 적용된 공간의 중심값이다.
    #
    # 따라서 바로 Z-score처럼 해석할 수 있다.

    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=features
    )


    print("\n[Cluster 중심 Z-score]")

    print(
        centers
        .round(2)
        .to_string()
    )


    # ==================================================
    # 7. Cluster별 높은 특징 / 낮은 특징 찾기
    # ==================================================

    print("\n" + "=" * 80)
    print("Cluster별 주요 특징")
    print("=" * 80)


    for cluster in range(k):

        print(f"\n--- Cluster {cluster} ---")


        cluster_center = centers.loc[cluster]


        # 높은 특징 TOP 5
        high_features = (
            cluster_center
            .sort_values(ascending=False)
            .head(5)
        )


        # 낮은 특징 TOP 5
        low_features = (
            cluster_center
            .sort_values(ascending=True)
            .head(5)
        )


        print("\n[평균보다 높은 특징 TOP 5]")

        for feature, value in high_features.items():

            print(
                f"{feature:<30} "
                f"{value:+.2f}"
            )


        print("\n[평균보다 낮은 특징 TOP 5]")

        for feature, value in low_features.items():

            print(
                f"{feature:<30} "
                f"{value:+.2f}"
            )


    # ==================================================
    # 8. Z-score 중심값 CSV 저장
    # ==================================================

    output_path = (
        f"machine_learning/"
        f"k{k}_cluster_zscore_centers.csv"
    )

    centers.to_csv(
        output_path,
        index_label="cluster",
        encoding="utf-8-sig"
    )


    print("\n저장 완료:", output_path)