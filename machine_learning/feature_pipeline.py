import numpy as np
import pandas as pd


# ============================================================
# K-Means에서 사용하는 최종 14개 행동 Feature
# ============================================================

STYLE_FEATURES = [
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
# 14개 Feature를 계산하기 위해 필요한 원본 컬럼
# ============================================================

REQUIRED_COLUMNS = [

    # 경기 정보
    "matchEndType",
    "matchId",
    "matchDate",

    # 유저 정보
    "ouid",
    "nickname",
    "division",

    # 점유 / 드리블
    "possession",
    "dribble",

    # 패스
    "passTry",
    "shortPassTry",
    "longPassTry",
    "bouncingLobPassTry",
    "drivenGroundPassTry",
    "throughPassTry",
    "lobbedThroughPassTry",

    # 슈팅
    "shootTotal",
    "shootHeading",
    "shootInPenalty",

    # 수비
    "tackleTry",
    "blockTry",
]


# ============================================================
# 안전한 나눗셈
# ============================================================

def safe_divide(numerator, denominator):

    return np.where(
        denominator > 0,
        numerator / denominator,
        0
    )


# ============================================================
# 유저 행동 Feature 생성
# ============================================================
#
# 원본 54개 컬럼 CSV
#       ↓
# 필요한 원본 컬럼만 사용
#       ↓
# 정상 종료 경기만 사용
#       ↓
# 최근 N경기 선택
#       ↓
# 최종 14개 행동 Feature 계산
#
# ============================================================

def build_behavior_features(
    raw_df,
    match_limit
):

    print()
    print("=" * 70)
    print("플레이스타일 행동 Feature 생성")
    print("=" * 70)

    df = raw_df.copy()

    print(
        "원본 데이터:",
        df.shape
    )

    # ========================================================
    # 1. 필요한 컬럼 확인
    # ========================================================

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("[오류] 필요한 컬럼이 없습니다.")

        for column in missing_columns:

            print(
                "-",
                column
            )

        raise ValueError(
            "원본 경기 CSV의 컬럼을 확인해주세요."
        )

    # ========================================================
    # 2. 필요한 컬럼만 사용
    # ========================================================

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # ========================================================
    # 3. 정상 종료 경기만 사용
    # ========================================================

    before_normal = len(df)

    df = df[
        df["matchEndType"] == 0
    ].copy()

    after_normal = len(df)

    print()
    print(
        "정상 종료 필터 전:",
        before_normal
    )

    print(
        "정상 종료 필터 후:",
        after_normal
    )

    print(
        "비정상 종료 제거:",
        before_normal - after_normal
    )

    # ========================================================
    # 4. 결측치 제거
    # ========================================================

    before_na = len(df)

    df = (
        df
        .dropna()
        .copy()
    )

    after_na = len(df)

    print()
    print(
        "결측치 제거:",
        before_na - after_na
    )

    # ========================================================
    # 5. 날짜 변환
    # ========================================================

    df["matchDate"] = pd.to_datetime(
        df["matchDate"],
        errors="coerce"
    )

    invalid_date_count = (
        df["matchDate"]
        .isna()
        .sum()
    )

    if invalid_date_count > 0:

        print(
            "날짜 변환 실패:",
            invalid_date_count
        )

        df = df[
            df["matchDate"].notna()
        ].copy()

    # ========================================================
    # 6. 동일 유저 + 동일 경기 중복 제거
    # ========================================================

    before_duplicate = len(df)

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

    after_duplicate = len(df)

    print(
        "중복 경기 제거:",
        before_duplicate - after_duplicate
    )

    # ========================================================
    # 7. 유저별 정상 경기 수
    # ========================================================

    match_counts = (
        df
        .groupby(
            "ouid"
        )
        .size()
    )

    print()
    print(
        "전체 유저 수:",
        df["ouid"].nunique()
    )

    print(
        f"{match_limit}경기 이상 유저:",
        (
            match_counts
            >= match_limit
        ).sum()
    )

    # ========================================================
    # 8. N경기 이상 유저만 선택
    # ========================================================

    valid_ouids = match_counts[
        match_counts >= match_limit
    ].index

    df = df[
        df["ouid"].isin(
            valid_ouids
        )
    ].copy()

    if df.empty:

        raise ValueError(
            f"정상 종료 경기 {match_limit}경기 이상인 "
            "유저가 없습니다."
        )

    # ========================================================
    # 9. 유저별 최근 N경기만 선택
    # ========================================================

    df = (
        df
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
            match_limit
        )
        .copy()
    )

    selected_counts = (
        df
        .groupby(
            "ouid"
        )
        .size()
    )

    if not (
        selected_counts == match_limit
    ).all():

        raise ValueError(
            "유저별 경기 수 선택에 문제가 있습니다."
        )

    print()
    print(
        f"최근 {match_limit}경기 선택 완료"
    )

    print(
        "선택된 유저:",
        df["ouid"].nunique()
    )

    print(
        "사용 경기:",
        len(df)
    )

    # ========================================================
    # 10. 유저 기본 정보
    # ========================================================

    user_info = (
        df
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

    # ========================================================
    # 11. 점유율 / 드리블 평균
    # ========================================================

    user_mean = (
        df
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

    # ========================================================
    # 12. 필요한 Raw Count 합계
    # ========================================================

    sum_columns = [
        "passTry",
        "shortPassTry",
        "longPassTry",
        "bouncingLobPassTry",
        "drivenGroundPassTry",
        "throughPassTry",
        "lobbedThroughPassTry",

        "shootTotal",
        "shootHeading",
        "shootInPenalty",

        "tackleTry",
        "blockTry",
    ]

    user_sum = (
        df
        .groupby(
            "ouid"
        )[sum_columns]
        .sum()
        .reset_index()
    )

    # ========================================================
    # 13. 사용 경기 수
    # ========================================================

    user_match_count = (
        df
        .groupby(
            "ouid"
        )
        .size()
        .rename(
            "match_count"
        )
        .reset_index()
    )

    # ========================================================
    # 14. 데이터 결합
    # ========================================================

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

    # ========================================================
    # 15. 최종 행동 Feature 14개 계산
    # ========================================================

    # ------------------------
    # 패스
    # ------------------------

    user_df["pass_per_match"] = safe_divide(
        user_df["passTry"],
        user_df["match_count"]
    )

    user_df["short_pass_rate"] = safe_divide(
        user_df["shortPassTry"],
        user_df["passTry"]
    )

    user_df["long_pass_rate"] = safe_divide(
        user_df["longPassTry"],
        user_df["passTry"]
    )

    user_df["bouncing_lob_pass_rate"] = safe_divide(
        user_df["bouncingLobPassTry"],
        user_df["passTry"]
    )

    user_df["driven_ground_pass_rate"] = safe_divide(
        user_df["drivenGroundPassTry"],
        user_df["passTry"]
    )

    user_df["through_pass_rate"] = safe_divide(
        user_df["throughPassTry"],
        user_df["passTry"]
    )

    user_df["lobbed_through_pass_rate"] = safe_divide(
        user_df["lobbedThroughPassTry"],
        user_df["passTry"]
    )

    # ------------------------
    # 슈팅
    # ------------------------

    user_df["shoot_per_match"] = safe_divide(
        user_df["shootTotal"],
        user_df["match_count"]
    )

    user_df["inside_penalty_rate"] = safe_divide(
        user_df["shootInPenalty"],
        user_df["shootTotal"]
    )

    user_df["heading_shoot_rate"] = safe_divide(
        user_df["shootHeading"],
        user_df["shootTotal"]
    )

    # ------------------------
    # 수비
    # ------------------------

    user_df["tackle_per_match"] = safe_divide(
        user_df["tackleTry"],
        user_df["match_count"]
    )

    user_df["block_per_match"] = safe_divide(
        user_df["blockTry"],
        user_df["match_count"]
    )

    # ========================================================
    # 16. 최종 결과
    # ========================================================

    final_columns = [
        "ouid",
        "nickname",
        "division",
        "match_count",
    ] + STYLE_FEATURES

    final_df = user_df[
        final_columns
    ].copy()

    # ========================================================
    # 17. 검증
    # ========================================================

    nan_count = (
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

    print()
    print(
        "NaN:",
        nan_count
    )

    print(
        "무한대:",
        inf_count
    )

    if nan_count > 0:

        raise ValueError(
            "최종 Feature에 NaN이 있습니다."
        )

    if inf_count > 0:

        raise ValueError(
            "최종 Feature에 무한대가 있습니다."
        )

    print()
    print(
        "최종 데이터:",
        final_df.shape
    )

    print(
        "행동 Feature 수:",
        len(STYLE_FEATURES)
    )

    return final_df


# ============================================================
# 직접 실행
# ============================================================

if __name__ == "__main__":

    INPUT_FILE = "match_data/match_team_data.csv"

    OUTPUT_FILE = (
    "machine_learning/data/"
    "user_behavior_features_70matches.csv")

    MATCH_LIMIT = 70

    raw_df = pd.read_csv(
        INPUT_FILE
    )

    final_df = build_behavior_features(
        raw_df,
        match_limit=MATCH_LIMIT
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 70)
    print("행동 Feature 생성 완료")
    print("=" * 70)

    print(
        "저장 위치:",
        OUTPUT_FILE
    )