import pandas as pd


# ==================================================
# 1. 데이터 불러오기
# ==================================================

file_path = "match_data/match_style_features.csv"

df = pd.read_csv(file_path)


# ==================================================
# 2. 기본 데이터 크기 확인
# ==================================================

print("=" * 60)
print("1. 기본 데이터 정보")
print("=" * 60)

print("전체 행 수:", len(df))
print("전체 컬럼 수:", len(df.columns))

print("고유 경기 수:", df["matchId"].nunique())
print("고유 유저 수:", df["ouid"].nunique())


# ==================================================
# 3. 결측치 확인
# ==================================================

print("\n" + "=" * 60)
print("2. 결측치 확인")
print("=" * 60)

missing = df.isnull().sum()

# 결측치가 있는 컬럼만 출력
print(missing[missing > 0])


# ==================================================
# 4. 중복 확인
# ==================================================

print("\n" + "=" * 60)
print("3. 경기-유저 중복 확인")
print("=" * 60)

duplicate_count = df.duplicated(
    subset=["matchId", "ouid"]
).sum()

print("중복 행 수:", duplicate_count)


# ==================================================
# 5. 주요 컬럼 0값 확인
# ==================================================

print("\n" + "=" * 60)
print("4. 주요 컬럼의 0값 개수")
print("=" * 60)

zero_check_columns = [
    "possession",
    "passTry",
    "shootTotal",
    "tackleTry",
    "blockTry"
]

for column in zero_check_columns:
    zero_count = (df[column] == 0).sum()

    print(
        f"{column}: "
        f"{zero_count}개 "
        f"({zero_count / len(df) * 100:.2f}%)"
    )


# ==================================================
# 6. 주요 컬럼 통계 확인
# ==================================================

print("\n" + "=" * 60)
print("5. 주요 컬럼 기초 통계")
print("=" * 60)

check_columns = [
    "possession",
    "dribble",
    "passTry",
    "passSuccess",
    "shortPassTry",
    "longPassTry",
    "throughPassTry",
    "shootTotal",
    "effectiveShootTotal",
    "shootInPenalty",
    "tackleTry",
    "tackleSuccess",
    "blockTry",
    "blockSuccess"
]

print(
    df[check_columns]
    .describe()
    .round(2)
    .T
)


# ==================================================
# 7. 말이 안 되는 값 확인
# ==================================================

print("\n" + "=" * 60)
print("6. 값 관계 이상 확인")
print("=" * 60)

print(
    "passSuccess > passTry:",
    (df["passSuccess"] > df["passTry"]).sum()
)

print(
    "effectiveShootTotal > shootTotal:",
    (df["effectiveShootTotal"] > df["shootTotal"]).sum()
)

print(
    "tackleSuccess > tackleTry:",
    (df["tackleSuccess"] > df["tackleTry"]).sum()
)

print(
    "blockSuccess > blockTry:",
    (df["blockSuccess"] > df["blockTry"]).sum()
)

print(
    "shortPassTry > passTry:",
    (df["shortPassTry"] > df["passTry"]).sum()
)

print(
    "longPassTry > passTry:",
    (df["longPassTry"] > df["passTry"]).sum()
)


# ==================================================
# 8. 점유율 범위 확인
# ==================================================

print("\n" + "=" * 60)
print("7. 점유율 범위 확인")
print("=" * 60)

print("최소 점유율:", df["possession"].min())
print("최대 점유율:", df["possession"].max())

invalid_possession = (
    (df["possession"] < 0)
    | (df["possession"] > 100)
).sum()

print("0~100 범위를 벗어난 데이터:", invalid_possession)


print("\n데이터 상태 검사 완료")