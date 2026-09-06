import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ==================================================
# 설정
# ==================================================

DATA_PATH = "match_data/user_style_features.csv"

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

K_VALUES = [2, 3, 4]

REFERENCE_RANDOM_STATE = 42
TEST_COUNT = 50
N_INIT = 10


# ==================================================
# 데이터 불러오기
# ==================================================

print("=" * 70)
print("K-Means 안정성 비교: K=2, K=3, K=4")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\n데이터 크기: {df.shape}")

print(f"\n사용 Feature 수: {len(FEATURES)}")
for feature in FEATURES:
    print(f" - {feature}")


# ==================================================
# Feature 선택
# ==================================================

X = df[FEATURES].copy()

print(f"\n결측치 개수: {X.isna().sum().sum()}")


# ==================================================
# 표준화
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n표준화 완료")


# ==================================================
# 결과 저장 리스트
# ==================================================

all_results = []


# ==================================================
# K별 안정성 테스트
# ==================================================

for k in K_VALUES:

    print("\n" + "=" * 70)
    print(f"K={k} 안정성 테스트")
    print("=" * 70)

    # --------------------------------------------------
    # 기준 모델
    # --------------------------------------------------

    reference_model = KMeans(
        n_clusters=k,
        random_state=REFERENCE_RANDOM_STATE,
        n_init=N_INIT,
    )

    reference_labels = reference_model.fit_predict(X_scaled)

    reference_silhouette = silhouette_score(
        X_scaled,
        reference_labels,
    )

    print("\n[기준 모델]")
    print(f"Random State: {REFERENCE_RANDOM_STATE}")
    print(f"Silhouette Score: {reference_silhouette:.4f}")

    unique, counts = np.unique(
        reference_labels,
        return_counts=True,
    )

    print("\n기준 모델 군집별 유저 수")

    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count}명")

    # --------------------------------------------------
    # Random State 반복
    # --------------------------------------------------

    k_results = []

    print(f"\n[K={k} 안정성 테스트 ({TEST_COUNT}회)]")

    for random_state in range(TEST_COUNT):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        ari = adjusted_rand_score(
            reference_labels,
            labels,
        )

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        row = {
            "k": k,
            "random_state": random_state,
            "ari_score": ari,
            "silhouette_score": silhouette,
        }

        all_results.append(row)
        k_results.append(row)

        print(
            f"random_state={random_state:2d} | "
            f"ARI={ari:.4f} | "
            f"silhouette={silhouette:.4f}"
        )

    # --------------------------------------------------
    # K별 결과 요약
    # --------------------------------------------------

    k_result_df = pd.DataFrame(k_results)

    print("\n" + "-" * 70)
    print(f"K={k} 결과 요약")
    print("-" * 70)

    print("\n[ARI]")
    print(k_result_df["ari_score"].describe())

    print("\n[Silhouette]")
    print(k_result_df["silhouette_score"].describe())

    ari_90 = (k_result_df["ari_score"] >= 0.90).sum()

    ari_80_90 = (
        (k_result_df["ari_score"] >= 0.80)
        & (k_result_df["ari_score"] < 0.90)
    ).sum()

    ari_60_80 = (
        (k_result_df["ari_score"] >= 0.60)
        & (k_result_df["ari_score"] < 0.80)
    ).sum()

    ari_under_60 = (
        k_result_df["ari_score"] < 0.60
    ).sum()

    print("\n[ARI 구간]")
    print(f"0.90 이상: {ari_90}회")
    print(f"0.80 ~ 0.90 미만: {ari_80_90}회")
    print(f"0.60 ~ 0.80 미만: {ari_60_80}회")
    print(f"0.60 미만: {ari_under_60}회")


# ==================================================
# 전체 결과 DataFrame
# ==================================================

result_df = pd.DataFrame(all_results)


# ==================================================
# K별 비교 요약표
# ==================================================

summary_rows = []

for k in K_VALUES:

    temp = result_df[
        result_df["k"] == k
    ]

    summary_rows.append(
        {
            "k": k,
            "ari_mean": temp["ari_score"].mean(),
            "ari_median": temp["ari_score"].median(),
            "ari_min": temp["ari_score"].min(),
            "ari_max": temp["ari_score"].max(),
            "ari_std": temp["ari_score"].std(),

            "silhouette_mean": temp["silhouette_score"].mean(),
            "silhouette_min": temp["silhouette_score"].min(),
            "silhouette_max": temp["silhouette_score"].max(),

            "ari_0.90_over": (
                temp["ari_score"] >= 0.90
            ).sum(),

            "ari_0.60_under": (
                temp["ari_score"] < 0.60
            ).sum(),
        }
    )

summary_df = pd.DataFrame(summary_rows)


# ==================================================
# 최종 비교 출력
# ==================================================

print("\n" + "=" * 70)
print("K=2 / K=3 / K=4 최종 비교")
print("=" * 70)

print(
    summary_df.round(4).to_string(
        index=False
    )
)


# ==================================================
# CSV 저장
# ==================================================

result_path = (
    "machine_learning/"
    "kmeans_stability_all_results.csv"
)

summary_path = (
    "machine_learning/"
    "kmeans_stability_summary.csv"
)

result_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig",
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 70)
print("안정성 비교 완료")
print("=" * 70)

print(f"\n상세 결과 저장: {result_path}")
print(f"요약 결과 저장: {summary_path}")