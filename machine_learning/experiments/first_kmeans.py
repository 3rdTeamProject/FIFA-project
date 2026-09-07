import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==================================================
# 1. 데이터 불러오기
# ==================================================

file_path = "match_data/match_ml_features.csv"

df = pd.read_csv(file_path)

print("전체 데이터 크기:", df.shape)


# ==================================================
# 2. K-Means에 실제 사용할 특징 선택
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
# 3. 스케일링
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==================================================
# 4. K = 2 ~ 8 비교
# ==================================================

print("\n" + "=" * 60)
print("K-Means Silhouette Score 비교")
print("=" * 60)

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )

# ==================================================
# 5. K = 2 ~ 5 상세 분석
# ==================================================

for k in range(2, 6):

    print("\n" + "=" * 80)
    print(f"K = {k} Cluster 상세 분석")
    print("=" * 80)

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    analysis_df = df.copy()
    analysis_df["cluster"] = labels


    # ==================================================
    # Cluster별 데이터 개수
    # ==================================================

    print("\n[Cluster별 데이터 개수]")

    cluster_counts = (
        analysis_df["cluster"]
        .value_counts()
        .sort_index()
    )

    print(cluster_counts.to_string())


    # ==================================================
    # Cluster별 비율
    # ==================================================

    print("\n[Cluster별 비율]")

    cluster_ratio = (
        analysis_df["cluster"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(cluster_ratio.round(2).to_string())


    # ==================================================
    # Cluster별 Feature 평균
    # ==================================================

    print("\n[Cluster별 Feature 평균]")

    cluster_means = (
        analysis_df
        .groupby("cluster")[features]
        .mean()
    )

    print(cluster_means.round(4).to_string())


    # ==================================================
    # 결과 CSV 저장
    # ==================================================

    output_path = f"machine_learning/k{k}_cluster_result.csv"

    analysis_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nK={k} 결과 저장 완료")
    print("저장 위치:", output_path)