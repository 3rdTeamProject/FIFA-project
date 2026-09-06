import pandas as pd
import numpy as np


# ==================================================
# 1. 파일 경로
# ==================================================

INPUT_FILE = "match_data/match_style_features.csv"
OUTPUT_FILE = "match_data/user_style_features.csv"

# 플레이스타일을 판단하기 위한 최소 경기 수
MIN_MATCHES = 10


# ==================================================
# 2. 데이터 불러오기
# ==================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("유저 단위 플레이스타일 Feature 생성")
print("=" * 70)

print("경기 단위 데이터:", df.shape)
print("전체 고유 OUID 수:", df["ouid"].nunique())


# ==================================================
# 3. 유저별 경기 수 계산
# ==================================================

match_counts = (
    df.groupby("ouid")
    .size()
    .rename("match_count")
)

print()
print("[유저별 경기 수 기준]")

print(
    f"{MIN_MATCHES}경기 이상 유저:",
    (match_counts >= MIN_MATCHES).sum()
)


# ==================================================
# 4. 최소 경기 수 이상인 유저만 선택
# ==================================================

valid_ouids = match_counts[
    match_counts >= MIN_MATCHES
].index

filtered_df = df[
    df["ouid"].isin(valid_ouids)
].copy()

print(
    "선택된 경기 데이터:",
    filtered_df.shape
)


# ==================================================
# 5. 안전한 나눗셈
# ==================================================

def safe_divide(numerator, denominator):

    return np.where(
        denominator > 0,
        numerator / denominator,
        0
    )


# ==================================================
# 6. 유저별 기본 정보
# ==================================================

# 닉네임은 변경될 수 있으므로
# 유저 식별 자체는 ouid 기준으로 한다.
#
# 같은 OUID에 여러 닉네임이 존재할 경우
# 데이터에서 마지막으로 등장한 닉네임을 표시용으로 사용한다.

user_info = (
    filtered_df
    .groupby("ouid", as_index=False)
    .agg(
        nickname=("nickname", "last"),
        division=("division", "last")
    )
)


# ==================================================
# 7. 유저별 평균 행동량
# ==================================================

user_mean = (
    filtered_df
    .groupby("ouid")
    .agg(
        possession=("possession", "mean"),
        dribble=("dribble", "mean")
    )
    .reset_index()
)


# ==================================================
# 8. 유저별 Raw Count 합계
# ==================================================

sum_columns = [

    # 패스
    "passTry",
    "passSuccess",

    "shortPassTry",
    "shortPassSuccess",

    "longPassTry",
    "longPassSuccess",

    "bouncingLobPassTry",
    "bouncingLobPassSuccess",

    "drivenGroundPassTry",
    "drivenGroundPassSuccess",

    "throughPassTry",
    "throughPassSuccess",

    "lobbedThroughPassTry",
    "lobbedThroughPassSuccess",

    # 슈팅
    "shootTotal",
    "effectiveShootTotal",

    "shootHeading",
    "shootInPenalty",
    "shootOutPenalty",

    # 수비
    "tackleTry",
    "tackleSuccess",

    "blockTry",
    "blockSuccess"
]


user_sum = (
    filtered_df
    .groupby("ouid")[sum_columns]
    .sum()
    .reset_index()
)


# ==================================================
# 9. 경기 수 추가
# ==================================================

user_match_count = (
    filtered_df
    .groupby("ouid")
    .size()
    .rename("match_count")
    .reset_index()
)


# ==================================================
# 10. 데이터 합치기
# ==================================================

user_df = user_info.merge(
    user_match_count,
    on="ouid",
    how="left"
)

user_df = user_df.merge(
    user_mean,
    on="ouid",
    how="left"
)

user_df = user_df.merge(
    user_sum,
    on="ouid",
    how="left"
)


# ==================================================
# 11. 플레이스타일 Feature 계산
# ==================================================

# --------------------------------------------------
# 패스
# --------------------------------------------------

# 경기당 평균 패스 시도
user_df["pass_per_match"] = safe_divide(
    user_df["passTry"],
    user_df["match_count"]
)


# 전체 패스 성공률
user_df["pass_success_rate"] = safe_divide(
    user_df["passSuccess"],
    user_df["passTry"]
)


# 짧은 패스 비율
user_df["short_pass_rate"] = safe_divide(
    user_df["shortPassTry"],
    user_df["passTry"]
)


# 긴 패스 비율
user_df["long_pass_rate"] = safe_divide(
    user_df["longPassTry"],
    user_df["passTry"]
)


# 바운싱 로빙 패스 비율
user_df["bouncing_lob_pass_rate"] = safe_divide(
    user_df["bouncingLobPassTry"],
    user_df["passTry"]
)


# 드리븐 그라운드 패스 비율
user_df["driven_ground_pass_rate"] = safe_divide(
    user_df["drivenGroundPassTry"],
    user_df["passTry"]
)


# 스루패스 비율
user_df["through_pass_rate"] = safe_divide(
    user_df["throughPassTry"],
    user_df["passTry"]
)


# 로빙 스루패스 비율
user_df["lobbed_through_pass_rate"] = safe_divide(
    user_df["lobbedThroughPassTry"],
    user_df["passTry"]
)


# --------------------------------------------------
# 슈팅
# --------------------------------------------------

# 경기당 평균 슈팅
user_df["shoot_per_match"] = safe_divide(
    user_df["shootTotal"],
    user_df["match_count"]
)


# 유효 슈팅 비율
user_df["effective_shoot_rate"] = safe_divide(
    user_df["effectiveShootTotal"],
    user_df["shootTotal"]
)


# 박스 안 슈팅 비율
user_df["inside_penalty_rate"] = safe_divide(
    user_df["shootInPenalty"],
    user_df["shootTotal"]
)


# 헤딩 슈팅 비율
user_df["heading_shoot_rate"] = safe_divide(
    user_df["shootHeading"],
    user_df["shootTotal"]
)


# --------------------------------------------------
# 수비
# --------------------------------------------------

# 경기당 평균 태클 시도
user_df["tackle_per_match"] = safe_divide(
    user_df["tackleTry"],
    user_df["match_count"]
)


# 태클 성공률
user_df["tackle_success_rate"] = safe_divide(
    user_df["tackleSuccess"],
    user_df["tackleTry"]
)


# 경기당 평균 블록 시도
user_df["block_per_match"] = safe_divide(
    user_df["blockTry"],
    user_df["match_count"]
)


# 블록 성공률
user_df["block_success_rate"] = safe_divide(
    user_df["blockSuccess"],
    user_df["blockTry"]
)


# ==================================================
# 12. 최종 저장할 컬럼
# ==================================================

final_columns = [

    # 식별 정보
    "ouid",
    "nickname",
    "division",
    "match_count",

    # 행동량
    "possession",
    "dribble",

    "pass_per_match",

    # 패스 선택 성향
    "short_pass_rate",
    "long_pass_rate",
    "bouncing_lob_pass_rate",
    "driven_ground_pass_rate",
    "through_pass_rate",
    "lobbed_through_pass_rate",

    # 슈팅 성향
    "shoot_per_match",
    "inside_penalty_rate",
    "heading_shoot_rate",

    # 수비 행동량
    "tackle_per_match",
    "block_per_match",

    # 성공률
    # K-Means에서는 나중에 제외할 수 있지만
    # 분석용으로 CSV에는 보관
    "pass_success_rate",
    "effective_shoot_rate",
    "tackle_success_rate",
    "block_success_rate"
]


final_df = user_df[
    final_columns
].copy()


# ==================================================
# 13. NaN / 무한대 확인
# ==================================================

print()
print("=" * 70)
print("최종 유저 Feature 확인")
print("=" * 70)

print(
    "최종 유저 수:",
    len(final_df)
)

print(
    "최종 데이터 크기:",
    final_df.shape
)

print(
    "NaN 개수:",
    final_df.isna().sum().sum()
)


numeric_df = final_df.select_dtypes(
    include=[np.number]
)

inf_count = np.isinf(
    numeric_df
).sum().sum()

print(
    "무한대 개수:",
    inf_count
)


print()
print("[경기 수 분포]")

print(
    final_df["match_count"]
    .describe()
)


# ==================================================
# 14. CSV 저장
# ==================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("유저 플레이스타일 Feature 저장 완료")
print("=" * 70)

print(
    "저장 위치:",
    OUTPUT_FILE
)

print(
    "최종 데이터 크기:",
    final_df.shape
)