import pandas as pd


# ==================================================
# 1. 원본 CSV 불러오기
# ==================================================

input_path = "match_data/match_team_data.csv"

df = pd.read_csv(input_path)

print("원본 데이터 크기:", df.shape)

# ==================================================
# 정상 종료 경기만 사용
# matchEndType == 0
# ==================================================

before_normal = len(df)

df = df[
    df["matchEndType"] == 0
].copy()

after_normal = len(df)

print("정상 종료 필터 전:", before_normal)
print("정상 종료 필터 후:", after_normal)
print("비정상 종료 제거:", before_normal - after_normal)


# ==================================================
# 2. 플레이 스타일 분석에 사용할 컬럼 선택
# ==================================================

selected_columns = [

    # 식별용 컬럼
    # 머신러닝 학습에는 안 넣지만
    # 나중에 어떤 경기/유저인지 확인하기 위해 유지
    # ------------------------------
    "matchId",
    "matchDate",   
    "ouid",
    "nickname",
    "division",

    # ------------------------------
    # 점유 / 드리블
    # ------------------------------
    "possession",
    "dribble",

    # ------------------------------
    # 패스
    # ------------------------------
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

    # ------------------------------
    # 슈팅
    # ------------------------------
    "shootTotal",
    "effectiveShootTotal",

    "shootHeading",
    "shootFreekick",

    "shootInPenalty",
    "shootOutPenalty",

    "shootPenaltyKick",

    # ------------------------------
    # 수비
    # ------------------------------
    "blockTry",
    "blockSuccess",

    "tackleTry",
    "tackleSuccess"
]


# ==================================================
# 3. 필요한 컬럼만 추출
# ==================================================

style_df = df[selected_columns].copy()

# ==================================================
# 결측치가 있는 행 제거
# ==================================================

before_count = len(style_df)

style_df = style_df.dropna().copy()

after_count = len(style_df)

print("결측치 제거 전:", before_count)
print("결측치 제거 후:", after_count)
print("제거된 행 수:", before_count - after_count)

# ==================================================
# 4. 새 CSV로 저장
# ==================================================

output_path = "match_data/match_style_features.csv"

style_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


print("머신러닝용 데이터 생성 완료")
print("저장 위치:", output_path)
print("데이터 크기:", style_df.shape)