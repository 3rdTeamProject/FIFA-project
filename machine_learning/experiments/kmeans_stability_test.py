import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


# ==================================================
# 설정
# ==================================================

DATA_PATH = "match_data/user_style_features_70matches.csv"

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

K_VALUES = [2, 3, 4, 5]

TEST_COUNT = 50
N_INIT = 10


# ==================================================
# 데이터 불러오기
# ==================================================

print("=" * 70)
print("K-Means 안정성 비교: K=2, K=3, K=4, K=5")
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

if X.isna().sum().sum() > 0:
    raise ValueError("Feature 데이터에 결측치가 있습니다.")


# ==================================================
# 표준화
# ==================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n표준화 완료")


# ==================================================
# 결과 저장 리스트
# ==================================================

all_run_results = []

all_pairwise_results = []

summary_rows = []


# ==================================================
# K별 안정성 테스트
# ==================================================

for k in K_VALUES:

    print("\n" + "=" * 70)
    print(f"K={k} 안정성 테스트")
    print("=" * 70)

    # --------------------------------------------------
    # 50회 K-Means 실행
    # --------------------------------------------------

    label_results = {}

    silhouette_scores = []

    print(f"\n[K={k} K-Means 반복 실행: {TEST_COUNT}회]")

    for random_state in range(TEST_COUNT):

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels,
        )

        # 각 random_state별 label 저장
        label_results[random_state] = labels

        silhouette_scores.append(silhouette)

        all_run_results.append(
            {
                "k": k,
                "random_state": random_state,
                "silhouette_score": silhouette,
            }
        )

        unique, counts = np.unique(
            labels,
            return_counts=True,
        )

        cluster_counts = dict(
            zip(
                unique,
                counts,
            )
        )

        print(
            f"random_state={random_state:2d} | "
            f"silhouette={silhouette:.4f} | "
            f"cluster_sizes={cluster_counts}"
        )

    # --------------------------------------------------
    # 모든 실행 결과끼리 ARI 비교
    # --------------------------------------------------

    print("\n[Pairwise ARI 계산]")

    pairwise_ari_scores = []

    random_states = list(
        label_results.keys()
    )

    for state_a, state_b in combinations(
        random_states,
        2,
    ):

        labels_a = label_results[state_a]
        labels_b = label_results[state_b]

        ari = adjusted_rand_score(
            labels_a,
            labels_b,
        )

        pairwise_ari_scores.append(ari)

        all_pairwise_results.append(
            {
                "k": k,
                "random_state_a": state_a,
                "random_state_b": state_b,
                "ari_score": ari,
            }
        )

    # --------------------------------------------------
    # 결과 DataFrame
    # --------------------------------------------------

    silhouette_series = pd.Series(
        silhouette_scores
    )

    ari_series = pd.Series(
        pairwise_ari_scores
    )

    # --------------------------------------------------
    # K별 결과 출력
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print(f"K={k} 결과 요약")
    print("-" * 70)

    print("\n[Silhouette]")
    print(
        silhouette_series.describe()
    )

    print("\n[Pairwise ARI]")
    print(
        ari_series.describe()
    )

    ari_90 = (
        ari_series >= 0.90
    ).sum()

    ari_80_90 = (
        (ari_series >= 0.80)
        & (ari_series < 0.90)
    ).sum()

    ari_60_80 = (
        (ari_series >= 0.60)
        & (ari_series < 0.80)
    ).sum()

    ari_under_60 = (
        ari_series < 0.60
    ).sum()

    total_pairs = len(
        ari_series
    )

    print("\n[ARI 구간]")
    print(f"전체 비교 쌍: {total_pairs}쌍")
    print(f"0.90 이상: {ari_90}쌍")
    print(f"0.80 ~ 0.90 미만: {ari_80_90}쌍")
    print(f"0.60 ~ 0.80 미만: {ari_60_80}쌍")
    print(f"0.60 미만: {ari_under_60}쌍")

    # --------------------------------------------------
    # 요약 저장
    # --------------------------------------------------

    summary_rows.append(
        {
            "k": k,

            "silhouette_mean":
                silhouette_series.mean(),

            "silhouette_median":
                silhouette_series.median(),

            "silhouette_min":
                silhouette_series.min(),

            "silhouette_max":
                silhouette_series.max(),

            "silhouette_std":
                silhouette_series.std(),

            "ari_mean":
                ari_series.mean(),

            "ari_median":
                ari_series.median(),

            "ari_min":
                ari_series.min(),

            "ari_max":
                ari_series.max(),

            "ari_std":
                ari_series.std(),

            "pair_count":
                total_pairs,

            "ari_0.90_over":
                ari_90,

            "ari_0.80_0.90":
                ari_80_90,

            "ari_0.60_0.80":
                ari_60_80,

            "ari_under_0.60":
                ari_under_60,
        }
    )


# ==================================================
# 전체 실행 결과 DataFrame
# ==================================================

run_result_df = pd.DataFrame(
    all_run_results
)

pairwise_result_df = pd.DataFrame(
    all_pairwise_results
)

summary_df = pd.DataFrame(
    summary_rows
)


# ==================================================
# 최종 비교 출력
# ==================================================

print("\n" + "=" * 70)
print("K=2 / K=3 / K=4 / K=5 최종 안정성 비교")
print("=" * 70)

print(
    summary_df.round(4).to_string(
        index=False
    )
)


# ==================================================
# CSV 저장 경로
# ==================================================

run_result_path = (
    "machine_learning/"
    "kmeans_stability_run_results.csv"
)

pairwise_result_path = (
    "machine_learning/"
    "kmeans_stability_pairwise_ari.csv"
)

summary_path = (
    "machine_learning/"
    "kmeans_stability_summary.csv"
)


# ==================================================
# CSV 저장
# ==================================================

run_result_df.to_csv(
    run_result_path,
    index=False,
    encoding="utf-8-sig",
)

pairwise_result_df.to_csv(
    pairwise_result_path,
    index=False,
    encoding="utf-8-sig",
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 완료 출력
# ==================================================

print("\n" + "=" * 70)
print("안정성 비교 완료")
print("=" * 70)

print(
    f"\n실행별 Silhouette 저장: "
    f"{run_result_path}"
)

print(
    f"Pairwise ARI 저장: "
    f"{pairwise_result_path}"
)

print(
    f"최종 요약 저장: "
    f"{summary_path}"
)