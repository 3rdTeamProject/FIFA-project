import pandas as pd
import numpy as np


# ==================================================
# 1. 파일 경로
# ==================================================

STYLE_FILE = "match_data/match_style_features.csv"
ORIGINAL_FILE = "match_data/match_team_data.csv"
OUTPUT_FILE = "match_data/match_ml_features.csv"


# ==================================================
# 2. 데이터 불러오기
# ==================================================

style_df = pd.read_csv(STYLE_FILE)
original_df = pd.read_csv(ORIGINAL_FILE)

print("=" * 70)
print("ML Feature 데이터 생성")
print("=" * 70)

print("\n[데이터 불러오기]")
print("Style 데이터:", style_df.shape)
print("원본 데이터:", original_df.shape)


# ==================================================
# 3. 원본 데이터에서 matchEndType 가져오기
# ==================================================

# 같은 경기의 같은 유저를 찾기 위해
# matchId + ouid를 사용한다.
#
# matchEndType:
# 0 = 정상종료
# 1 = 몰수승
# 2 = 몰수패

end_type_df = original_df[
    [
        "matchId",
        "ouid",
        "matchEndType"
    ]
].copy()


# 혹시 모를 중복 방지
end_type_df = end_type_df.drop_duplicates(
    subset=["matchId", "ouid"]
)


# ==================================================
# 4. Style 데이터와 matchEndType 연결
# ==================================================

df = style_df.merge(
    end_type_df,
    on=["matchId", "ouid"],
    how="left"
)

print("\n[matchEndType 연결 후]")
print("데이터 크기:", df.shape)


# ==================================================
# 5. matchEndType 누락 확인
# ==================================================

missing_end_type = df["matchEndType"].isna().sum()

print("matchEndType 누락:", missing_end_type)


# ==================================================
# 6. 정상종료 경기만 남기기
# ==================================================

print("\n[경기 종료 타입 분포 - 필터링 전]")

print(
    df["matchEndType"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


before_count = len(df)


# 0 = 정상종료
df = df[
    df["matchEndType"] == 0
].copy()


after_count = len(df)

removed_count = before_count - after_count


print("\n[정상종료 경기 필터링]")

print("필터링 전:", before_count)
print("정상종료:", after_count)
print("제외된 몰수 경기:", removed_count)

print(
    "제외 비율:",
    f"{removed_count / before_count * 100:.2f}%"
)


# ==================================================
# 7. 안전한 나눗셈 함수
# ==================================================

def safe_divide(numerator, denominator):
    """
    denominator가 0보다 크면 나눗셈을 수행하고,
    denominator가 0이면 0으로 처리한다.
    """

    return np.where(
        denominator > 0,
        numerator / denominator,
        0
    )


# ==================================================
# 8. 새로운 ML Feature 생성
# ==================================================

ml_df = pd.DataFrame()


# --------------------------------------------------
# 식별용 컬럼
# --------------------------------------------------

ml_df["matchId"] = df["matchId"]
ml_df["ouid"] = df["ouid"]
ml_df["nickname"] = df["nickname"]
ml_df["division"] = df["division"]


# --------------------------------------------------
# 점유 / 개인 전개
# --------------------------------------------------

ml_df["possession"] = df["possession"]

ml_df["dribble"] = df["dribble"]


# --------------------------------------------------
# 패스 관련 Feature
# --------------------------------------------------

# 점유율 대비 패스 시도량
ml_df["pass_per_possession"] = safe_divide(
    df["passTry"],
    df["possession"]
)


# 패스 성공률
ml_df["pass_success_rate"] = safe_divide(
    df["passSuccess"],
    df["passTry"]
)


# 짧은 패스 비율
ml_df["short_pass_rate"] = safe_divide(
    df["shortPassTry"],
    df["passTry"]
)


# 긴 패스 비율
ml_df["long_pass_rate"] = safe_divide(
    df["longPassTry"],
    df["passTry"]
)


# 바운싱 로빙 패스 비율
ml_df["bouncing_lob_pass_rate"] = safe_divide(
    df["bouncingLobPassTry"],
    df["passTry"]
)


# 드리븐 그라운드 패스 비율
ml_df["driven_ground_pass_rate"] = safe_divide(
    df["drivenGroundPassTry"],
    df["passTry"]
)


# 스루패스 비율
ml_df["through_pass_rate"] = safe_divide(
    df["throughPassTry"],
    df["passTry"]
)


# 로빙 스루패스 비율
ml_df["lobbed_through_pass_rate"] = safe_divide(
    df["lobbedThroughPassTry"],
    df["passTry"]
)


# --------------------------------------------------
# 슈팅 관련 Feature
# --------------------------------------------------

# 점유율 대비 슈팅 횟수
ml_df["shoot_per_possession"] = safe_divide(
    df["shootTotal"],
    df["possession"]
)


# 슈팅 시도 여부
ml_df["shoot_attempted"] = (
    df["shootTotal"] > 0
).astype(int)


# 유효 슈팅 비율
ml_df["effective_shoot_rate"] = safe_divide(
    df["effectiveShootTotal"],
    df["shootTotal"]
)


# 페널티 박스 안 슈팅 비율
ml_df["inside_penalty_rate"] = safe_divide(
    df["shootInPenalty"],
    df["shootTotal"]
)


# 헤딩 슈팅 비율
ml_df["heading_shoot_rate"] = safe_divide(
    df["shootHeading"],
    df["shootTotal"]
)


# --------------------------------------------------
# 수비 관련 Feature
# --------------------------------------------------

# 점유율 대비 태클 시도
ml_df["tackle_per_possession"] = safe_divide(
    df["tackleTry"],
    df["possession"]
)


# 태클 시도 여부
ml_df["tackle_attempted"] = (
    df["tackleTry"] > 0
).astype(int)


# 태클 성공률
ml_df["tackle_success_rate"] = safe_divide(
    df["tackleSuccess"],
    df["tackleTry"]
)


# 점유율 대비 블록 시도
ml_df["block_per_possession"] = safe_divide(
    df["blockTry"],
    df["possession"]
)


# 블록 시도 여부
ml_df["block_attempted"] = (
    df["blockTry"] > 0
).astype(int)


# 블록 성공률
ml_df["block_success_rate"] = safe_divide(
    df["blockSuccess"],
    df["blockTry"]
)


# ==================================================
# 9. NaN / 무한대 확인
# ==================================================

print("\n" + "=" * 70)
print("생성된 ML Feature 확인")
print("=" * 70)

print("ML 데이터 크기:", ml_df.shape)

print(
    "NaN 개수:",
    ml_df.isna().sum().sum()
)


numeric_df = ml_df.select_dtypes(
    include=[np.number]
)

inf_count = np.isinf(
    numeric_df
).sum().sum()

print("무한대 개수:", inf_count)


# ==================================================
# 10. 시도하지 않은 경기 수 확인
# ==================================================

print("\n[시도 여부 확인]")

print(
    "슈팅 0회:",
    (ml_df["shoot_attempted"] == 0).sum()
)

print(
    "태클 0회:",
    (ml_df["tackle_attempted"] == 0).sum()
)

print(
    "블록 0회:",
    (ml_df["block_attempted"] == 0).sum()
)


# ==================================================
# 11. 최종 컬럼 확인
# ==================================================

print("\n[최종 컬럼]")

for column in ml_df.columns:
    print(column)


# ==================================================
# 12. CSV 저장
# ==================================================

ml_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("ML Feature CSV 저장 완료")
print("=" * 70)

print("저장 위치:", OUTPUT_FILE)
print("최종 데이터 크기:", ml_df.shape)