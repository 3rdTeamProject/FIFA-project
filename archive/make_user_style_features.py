import pandas as pd
import numpy as np


# ==================================================
# 1. 파일 경로 / 설정
# ==================================================

INPUT_FILE = "match_data/match_style_features.csv"
INPUT_FILE = "match_data/user_style_features_70matches.csv"

# 플레이스타일 분석 대상이 되기 위한 최소 보유 경기 수
MIN_MATCHES = 70

# 실제 플레이스타일 계산에 사용할 경기 수
MATCHES_PER_USER = 70


# ==================================================
# 2. 데이터 불러오기
# ==================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("유저 단위 플레이스타일 Feature 생성")
print("=" * 70)

print(
    "경기 단위 데이터:",
    df.shape
)

print(
    "전체 고유 OUID 수:",
    df["ouid"].nunique()
)


# ==================================================
# 3. 필수 컬럼 확인
# ==================================================

required_columns = [
    "ouid",
    "nickname",
    "matchDate",
    "division",
    "possession",
    "dribble",
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
    "shootTotal",
    "effectiveShootTotal",
    "shootHeading",
    "shootInPenalty",
    "shootOutPenalty",
    "tackleTry",
    "tackleSuccess",
    "blockTry",
    "blockSuccess"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print()
    print(
        "[오류] 필요한 컬럼이 없습니다:"
    )

    for column in missing_columns:

        print(
            "-",
            column
        )

    raise ValueError(
        "match_style_features.csv의 "
        "컬럼을 확인해주세요."
    )


# ==================================================
# 4. 날짜 타입 변환
# ==================================================
#
# 최근 70경기를 선택하기 위해
# matchDate를 날짜형으로 변환한다.
# ==================================================

df["matchDate"] = pd.to_datetime(
    df["matchDate"],
    errors="coerce"
)


invalid_date_count = (
    df["matchDate"]
    .isna()
    .sum()
)


print()
print(
    "matchDate 변환 실패:",
    invalid_date_count
)


# 날짜가 없는 데이터가 있다면
# 최근 경기 판단이 불가능하므로 제외
if invalid_date_count > 0:

    print(
        f"날짜가 없는 {invalid_date_count}행을 제외합니다."
    )

    df = df[
        df["matchDate"].notna()
    ].copy()


# ==================================================
# 5. 동일 유저 + 동일 경기 중복 제거
# ==================================================
#
# 혹시 이전 데이터 누적 과정에서
# 같은 경기가 중복으로 들어간 경우를 대비한다.
#
# matchId가 존재하면
# ouid + matchId 기준 중복 제거
# ==================================================

if "matchId" in df.columns:

    before_count = len(df)

    df = (
        df
        .sort_values(
            "matchDate"
        )
        .drop_duplicates(
            subset=[
                "ouid",
                "matchId"
            ],
            keep="last"
        )
        .copy()
    )

    after_count = len(df)

    print()
    print(
        "중복 경기 제거:",
        before_count - after_count,
        "행"
    )


# ==================================================
# 6. 유저별 보유 경기 수 계산
# ==================================================

match_counts = (
    df
    .groupby("ouid")
    .size()
    .rename("original_match_count")
)


print()
print("=" * 70)
print("유저별 경기 수 확인")
print("=" * 70)


print(
    f"{MIN_MATCHES}경기 이상 유저:",
    (match_counts >= MIN_MATCHES).sum()
)

print(
    f"{MIN_MATCHES}경기 미만 유저:",
    (match_counts < MIN_MATCHES).sum()
)


print()
print("[전체 유저 경기 수 분포]")

print(
    match_counts.describe()
)


# ==================================================
# 7. 70경기 이상인 유저만 선택
# ==================================================

valid_ouids = match_counts[
    match_counts >= MIN_MATCHES
].index


filtered_df = df[
    df["ouid"].isin(
        valid_ouids
    )
].copy()


print()
print(
    "70경기 이상 유저의 전체 경기 데이터:",
    filtered_df.shape
)

print(
    "선택된 유저 수:",
    filtered_df["ouid"].nunique()
)


# ==================================================
# 8. ★ 유저별 최근 70경기만 선택
# ==================================================
#
# 예:
#
# A 100경기 → 70경기 사용
# B 137경기 → 최근 70경기 사용
# C 250경기 → 최근 70경기 사용
#
# 날짜 내림차순으로 정렬한 뒤
# 유저별 상위 100개를 선택한다.
# ==================================================

filtered_df = (
    filtered_df
    .sort_values(
        [
            "ouid",
            "matchDate"
        ],
        ascending=[
            True,
            False
        ]
    )
    .groupby(
        "ouid",
        group_keys=False
    )
    .head(
        MATCHES_PER_USER
    )
    .copy()
)


print()
print("=" * 70)
print("유저별 최근 70경기 선택 완료")
print("=" * 70)

print(
    "선택된 데이터:",
    filtered_df.shape
)

print(
    "선택된 유저 수:",
    filtered_df["ouid"].nunique()
)


# ==================================================
# 9. 최근 70경기 선택 결과 검증
# ==================================================

selected_match_counts = (
    filtered_df
    .groupby("ouid")
    .size()
)


invalid_users = selected_match_counts[
    selected_match_counts != MATCHES_PER_USER
]


if not invalid_users.empty:

    print()
    print(
        "[오류] 70경기가 아닌 유저가 있습니다."
    )

    print(
        invalid_users
    )

    raise ValueError(
        "유저별 최근 70경기 선택에 문제가 있습니다."
    )


print(
    "모든 선택 유저가",
    MATCHES_PER_USER,
    "경기씩 보유하고 있습니다."
)


# ==================================================
# 10. 안전한 나눗셈
# ==================================================

def safe_divide(
    numerator,
    denominator
):

    return np.where(
        denominator > 0,
        numerator / denominator,
        0
    )


# ==================================================
# 11. 유저별 기본 정보
# ==================================================
#
# OUID를 실제 유저 식별자로 사용한다.
#
# 최근 경기부터 정렬된 상태이므로
# nickname / division은 first를 사용하면
# 가장 최근 경기의 값을 가져올 수 있다.
# ==================================================

user_info = (
    filtered_df
    .groupby(
        "ouid",
        as_index=False
    )
    .agg(
        nickname=(
            "nickname",
            "first"
        ),
        division=(
            "division",
            "first"
        )
    )
)


# ==================================================
# 12. 유저별 평균 행동량
# ==================================================

user_mean = (
    filtered_df
    .groupby(
        "ouid"
    )
    .agg(
        possession=(
            "possession",
            "mean"
        ),
        dribble=(
            "dribble",
            "mean"
        )
    )
    .reset_index()
)


# ==================================================
# 13. 유저별 Raw Count 합계
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
    .groupby(
        "ouid"
    )[sum_columns]
    .sum()
    .reset_index()
)


# ==================================================
# 14. 사용 경기 수
# ==================================================

user_match_count = (
    filtered_df
    .groupby(
        "ouid"
    )
    .size()
    .rename(
        "match_count"
    )
    .reset_index()
)


# ==================================================
# 15. 원래 보유하고 있던 경기 수도 저장
# ==================================================
#
# 분석에는 최근 70경기만 사용하지만
# 해당 유저가 원래 몇 경기를 보유했는지는
# 확인용으로 보관할 수 있다.
# ==================================================

original_match_count_df = (
    match_counts
    .reset_index()
)


# ==================================================
# 16. 데이터 합치기
# ==================================================

user_df = user_info.merge(
    user_match_count,
    on="ouid",
    how="left"
)


user_df = user_df.merge(
    original_match_count_df,
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
# 17. 플레이스타일 Feature 계산
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
# 18. 최종 저장 컬럼
# ==================================================

final_columns = [

    # 식별 정보
    "ouid",
    "nickname",
    "division",

    # 실제 분석에 사용한 경기 수
    "match_count",

    # 해당 유저가 원래 보유했던 경기 수
    "original_match_count",

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
    "pass_success_rate",
    "effective_shoot_rate",
    "tackle_success_rate",
    "block_success_rate"
]


final_df = user_df[
    final_columns
].copy()


# ==================================================
# 19. NaN / 무한대 확인
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
    final_df
    .isna()
    .sum()
    .sum()
)


numeric_df = final_df.select_dtypes(
    include=[
        np.number
    ]
)


inf_count = (
    np.isinf(
        numeric_df
    )
    .sum()
    .sum()
)


print(
    "무한대 개수:",
    inf_count
)


# ==================================================
# 20. 최종 경기 수 검증
# ==================================================

print()
print("[분석에 사용된 경기 수]")

print(
    final_df[
        "match_count"
    ]
    .value_counts()
    .sort_index()
)


if not (
    final_df["match_count"]
    == MATCHES_PER_USER
).all():

    raise ValueError(
        "70경기가 아닌 유저가 "
        "최종 데이터에 존재합니다."
    )


print()
print(
    "✅ 모든 최종 유저의 분석 경기 수:",
    MATCHES_PER_USER
)


# ==================================================
# 21. 원래 보유 경기 수 확인
# ==================================================

print()
print("[원래 보유 경기 수 분포]")

print(
    final_df[
        "original_match_count"
    ]
    .describe()
)


# ==================================================
# 22. CSV 저장
# ==================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 23. 최종 결과
# ==================================================

print()
print("=" * 70)
print("유저 플레이스타일 Feature 저장 완료")
print("=" * 70)


print(
    "저장 위치:",
    OUTPUT_FILE
)


print(
    "최종 유저 수:",
    len(final_df)
)


print(
    "유저당 사용 경기 수:",
    MATCHES_PER_USER
)


print(
    "최종 데이터 크기:",
    final_df.shape
)


print()
print("[최종 유저 목록]")


print(
    final_df[
        [
            "nickname",
            "match_count",
            "original_match_count"
        ]
    ]
    .sort_values(
        "nickname"
    )
    .to_string(
        index=False
    )
)


print("=" * 70)