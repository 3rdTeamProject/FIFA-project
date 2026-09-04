import pandas as pd
import numpy as np


# ==================================================
# 1. ML용 데이터 불러오기
# ==================================================

file_path = "match_data/match_ml_features.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("1. 기본 정보")
print("=" * 60)

print("데이터 크기:", df.shape)
print("고유 경기 수:", df["matchId"].nunique())
print("고유 유저 수:", df["ouid"].nunique())


# ==================================================
# 2. 실제 K-Means에 사용할 특징
# ==================================================

features = [
    "possession",
    "dribble",

    "pass_per_possession",
    "pass_success_rate",
    "short_pass_rate",
    "long_pass_rate",
    "bouncing_lob_pass_rate",
    "driven_ground_pass_rate",
    "through_pass_rate",
    "lobbed_through_pass_rate",

    "shoot_per_possession",
    "effective_shoot_rate",
    "inside_penalty_rate",
    "heading_shoot_rate",

    "tackle_per_possession",
    "tackle_success_rate",
    "block_per_possession",
    "block_success_rate"
]


# ==================================================
# 3. 결측치 확인
# ==================================================

print("\n" + "=" * 60)
print("2. 결측치 확인")
print("=" * 60)

missing = df[features].isnull().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("결측치 없음")


# ==================================================
# 4. 무한대 값 확인
# ==================================================

print("\n" + "=" * 60)
print("3. 무한대 값 확인")
print("=" * 60)

inf_count = np.isinf(
    df[features].to_numpy()
).sum()

print("inf / -inf 개수:", inf_count)


# ==================================================
# 5. 특징별 기초 통계
# ==================================================

print("\n" + "=" * 60)
print("4. 특징별 기초 통계")
print("=" * 60)

print(
    df[features]
    .describe()
    .round(3)
    .T
)


# ==================================================
# 6. 0~1 비율 특징 범위 검사
# ==================================================

print("\n" + "=" * 60)
print("5. 비율 특징 범위 검사")
print("=" * 60)

rate_features = [
    "pass_success_rate",
    "short_pass_rate",
    "long_pass_rate",
    "bouncing_lob_pass_rate",
    "driven_ground_pass_rate",
    "through_pass_rate",
    "lobbed_through_pass_rate",
    "effective_shoot_rate",
    "inside_penalty_rate",
    "heading_shoot_rate",
    "tackle_success_rate",
    "block_success_rate"
]

for column in rate_features:

    invalid = (
        (df[column] < 0)
        | (df[column] > 1)
    ).sum()

    print(
        f"{column}: "
        f"최소={df[column].min():.3f}, "
        f"최대={df[column].max():.3f}, "
        f"범위 밖={invalid}"
    )


print("\nML 특징 데이터 검사 완료")